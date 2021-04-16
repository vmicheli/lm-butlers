import logging
import time
import os
import json

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import RandomSampler, DataLoader

import wandb

import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

from utils import DataCollatorForActionModeling, DemonstrationsDataset, set_seed


@dataclass
class TrainerArguments:
    tokenizer_path: str = 'gpt2-medium'
    model_path: str = 'gpt2-medium'

    lr: float = 1e-5
    train_batch_size: int = 1
    grad_acc_steps: int = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    epochs: int = 10
    num_episodes_per_epoch: int = 100
    num_eval_episodes: int = 140

    do_train: bool = True
    do_eval: bool = True
    eval_before_training: bool = True
    train_data_source: str = 'train'
    eval_data_source: str = 'eval_in_distribution'
    max_steps_per_training_episode: int = 30
    max_steps_per_eval_episode: int = 50
    max_context_size: int = 1000
    max_action_length: int = 20
    unstuck_eval: bool = False
    unstuck_num: int = 10
    unstuck_beams: int = 10
    humans_anns_ratio: float = 0.0

    render: bool = False
    wandb_logging: bool = True
    save_model_dir: Optional[str] = None
    save_checkpoints: bool = False
    start_checkpointing_on_epoch: int = 0
    seed: int = 42


class Trainer:
    def __init__(self, config):
        self.args = TrainerArguments(**config['trainer_args'])

        if self.args.wandb_logging:
            wandb.init(project="alfworld-lm", config=vars(self.args))

        set_seed(self.args.seed)

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token # gpt has no pad token
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path)

        self.modality_start_tokens = self.tokenizer(' [STATE] ')['input_ids']
        self.modality_end_tokens = self.tokenizer(' [ACTION]')['input_ids']
        self.action_end_token = self.tokenizer(' [')['input_ids'][0]

        # setup environment
        config['env']['goal_desc_human_anns_prob'] = self.args.humans_anns_ratio

        if self.args.do_train:
            config['dagger']['training']['max_nb_steps_per_episode'] = self.args.max_steps_per_training_episode
            self.train_env = getattr(environment, 'AlfredTWEnv')(config, train_eval=self.args.train_data_source)
            self.train_env = self.train_env.init_env(batch_size=1)
            self.train_env.seed(self.args.seed) # by seeding here, task tuples are not the same from one epoch to another

        if self.args.do_eval:
            config['dagger']['training']['max_nb_steps_per_episode'] = self.args.max_steps_per_eval_episode
            self.eval_env = getattr(environment, 'AlfredTWEnv')(config, train_eval=self.args.eval_data_source)
            self.eval_env = self.eval_env.init_env(batch_size=1)

        # setup device
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.args.do_train:
            self.optimizer = self._create_optimizer()

    def _save_model_and_tokenizer(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(optimizer_grouped_parameters, lr=self.args.lr)

    def _update_context(self, context, to_append, is_obs):
        return context + self.modality_start_tokens + to_append + self.modality_end_tokens if is_obs else context + to_append

    def _format_first_obs(self, obs):
        return obs.split('room.')[1].replace('\n', ' ', 1).replace('\n', '').strip()

    def _process_eval_context(self, context):
        first_obs_end = context.index(44710)
        first_obs = context[:first_obs_end-1]

        stripped_context = context[len(first_obs):]
        stripped_context = stripped_context[-(self.args.max_context_size - len(first_obs)):]

        return first_obs + stripped_context

    def _forward_batch(self, batch):
        for key, value in batch.items():
            batch[key] = value.to(self.device)

        outputs = self.model(**batch)

        return outputs

    def _fit(self, dems):
        self.model.train()
        self.model.zero_grad()

        data_collator = DataCollatorForActionModeling(tokenizer=self.tokenizer, mlm=False)
        dataset = DemonstrationsDataset(dems)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.train_batch_size, collate_fn=data_collator)

        for step, batch in enumerate(dataloader):
            loss = self._forward_batch(batch).loss
            loss = loss / self.args.grad_acc_steps
            loss.backward()

            if (step + 1) % self.args.grad_acc_steps == 0:
                if self.args.max_grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()

        self.model.eval()

    def _step_unstuck(self, context, steps):
        actions = self.model.generate(torch.tensor(context).unsqueeze(0).to(self.device),
                                      num_return_sequences=self.args.unstuck_num, num_beams=self.args.unstuck_beams,
                                      max_length=len(context)+self.args.max_action_length,
                                      eos_token_id=self.action_end_token)[:, len(context):-1].tolist()

        actions = [[e for e in action if e != self.action_end_token] for action in actions]

        decoded_action = self.tokenizer.decode(actions[0])[1:]

        obs, _, dones, infos = self.eval_env.step([decoded_action])
        obs = obs[0]
        done = dones[0]
        reward = float(infos['won'][0])
        steps += 1

        stuck_counter = 1
        while (obs == 'Nothing happens.') and (stuck_counter < len(actions)) and (not done):
            decoded_action = self.tokenizer.decode(actions[stuck_counter])[1:]

            obs, _, dones, infos = self.eval_env.step([decoded_action])
            obs = obs[0]
            done = dones[0]
            reward = float(infos['won'][0])
            steps += 1
            stuck_counter += 1

        action = actions[stuck_counter-1] if (stuck_counter < len(actions)) else actions[0]

        return action, obs, reward, done, steps

    def _step(self, context, steps):
        action = self.model.generate(torch.tensor(context).unsqueeze(0).to(self.device),
                                     max_length=len(context)+self.args.max_action_length,
                                     eos_token_id=self.action_end_token)[0, len(context):-1].tolist()

        decoded_action = self.tokenizer.decode(action)[1:]

        obs, _, dones, infos = self.eval_env.step([decoded_action])
        obs = obs[0]
        done = dones[0]
        reward = float(infos['won'][0])
        steps += 1

        return action, obs, reward, done, steps

    def eval(self):
        print("Evaluation...")
        start_time = time.time()

        self.eval_env.seed(self.args.seed)
        self.model.eval()

        ep_successes = []
        ep_lengths = []

        for episode_no in range(self.args.num_eval_episodes):
            print(f'Evaluating episode {episode_no}')

            # reset episode specific data
            obs, _ = self.eval_env.reset()
            done = False
            context = []

            obs = obs[0]
            obs = self._format_first_obs(obs)[:-1]
            steps = 0

            if self.args.render:
                print(f"Obs: {obs}")

            while not done:
                context = self._update_context(context, self.tokenizer(obs)['input_ids'], True)

                context = self._process_eval_context(context)

                action, obs, reward, done, steps = self._step_unstuck(context, steps) if self.args.unstuck_eval else self._step(context, steps)

                context = self._update_context(context, action, False)

                if self.args.render:
                    print("Action: {}, Obs: {}, Reward: {}".format(self.tokenizer.decode(action)[1:], obs, reward))

            # episode statistics
            ep_successes.append(reward)
            ep_lengths.append(steps)

        mean_ep_success_rate = np.mean(ep_successes)
        mean_ep_length = np.mean(ep_lengths)
        print(f'Mean episode success rate: {mean_ep_success_rate:.3f}   Mean episode length: {mean_ep_length:.3f}')
        print(f"Evaluation took: {time.time() - start_time:.2f}")

        metrics_eval = {'mean_eval_ep_success_rate': mean_ep_success_rate, 'mean_eval_ep_length': mean_ep_length}

        return metrics_eval

    def train(self):
        print("Training...")
        start_time = time.time()

        dems = []

        ep_returns = []
        ep_lengths = []

        for episode_no in range(self.args.num_episodes_per_epoch):
            print(f'Training episode {episode_no}')

            # reset episode specific data
            obs, _ = self.train_env.reset()
            done = False
            context = []
            ep_rews = []

            obs = obs[0]
            obs = self._format_first_obs(obs)
            obs_ids = self.tokenizer(obs)['input_ids']

            if self.args.render:
                print(f"Obs: {obs}")

            while (not done) and (len(context)+len(obs_ids)+self.args.max_action_length < self.args.max_context_size):

                context = self._update_context(context, obs_ids, True)

                action = self.model.generate(torch.tensor(context).unsqueeze(0).to(self.device),
                                             max_length=len(context)+self.args.max_action_length,
                                             do_sample=True, eos_token_id=self.action_end_token)[0, len(context):-1].tolist()

                decoded_action = self.tokenizer.decode(action)[1:]

                obs, _, dones, infos = self.train_env.step([decoded_action])
                obs = obs[0]
                obs_ids = self.tokenizer(obs)['input_ids']
                done = dones[0]
                reward = float(infos['won'][0])

                context = self._update_context(context, action, False)

                ep_rews.append(reward)

                if self.args.render:
                    print("Action: {}, Obs: {}, Reward: {}".format(decoded_action, obs, reward))

            # episode statistics
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)

            # store episode if successful
            if reward:
                dems.append(context)

        # fit on demonstrations
        self._fit(dems)

        mean_ep_return = np.mean(ep_returns)
        mean_ep_length = np.mean(ep_lengths)
        print(f'Mean episode return: {mean_ep_return:.3f}   Mean episode length: {mean_ep_length:.3f}')
        print(f"Training took: {time.time() - start_time:.2f}")

        metrics_train = {'mean_train_ep_return': mean_ep_return, 'mean_train_ep_length': mean_ep_length}

        return metrics_train

    def run(self):
        total_time = time.time()
        metrics = []

        if self.args.eval_before_training:
            metrics_eval = self.eval()
            wandb.log(metrics_eval)

        for epoch in range(self.args.epochs):
            print(f"Epoch: {epoch}")
            metrics_epoch = {}

            if self.args.do_train:
                metrics_train_epoch = self.train()
                metrics_epoch = {**metrics_epoch, **metrics_train_epoch}

            if self.args.do_eval:
                metrics_eval_epoch = self.eval()
                metrics_epoch = {**metrics_epoch, **metrics_eval_epoch}

            metrics.append(metrics_epoch)

            if self.args.wandb_logging:
                wandb.log(metrics_epoch)

            if self.args.save_model_dir is not None and self.args.save_checkpoints and (epoch >= self.args.start_checkpointing_on_epoch):
                checkpoint_dir = os.path.join(self.args.save_model_dir, 'checkpoints', f'epoch_{epoch}')
                print(f"Saving checkpoint under {checkpoint_dir} !")
                self._save_model_and_tokenizer(checkpoint_dir)

        if self.args.save_model_dir is not None:
            print(f"Saving model under {self.args.save_model_dir} !")
            self._save_model_and_tokenizer(self.args.save_model_dir)
            with open(os.path.join(self.args.save_model_dir, 'metrics'), "w") as handle:
                json.dump(metrics, handle)

        print(f"Run complete! It took: {(time.time() - total_time):.2f}")


if __name__ == '__main__':
    logging.getLogger("transformers").setLevel(logging.ERROR)

    configuration = generic.load_config()
    trainer = Trainer(configuration)
    trainer.run()
