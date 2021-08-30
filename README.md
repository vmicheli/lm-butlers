# Language Models are Few-Shot Butlers

This repository contains the code to reproduce the experiments in [Language Models are Few-Shot Butlers](https://arxiv.org/abs/2104.07972).

## Installation

#### ALFWorld
```
$ git clone https://github.com/alfworld/alfworld.git
$ cd alfworld
$ git checkout c0501c807fa77dd25ecfc86e92cd782226f5c74f
$ sed -i '/torch/d' requirements.txt 
$ pip3 install -r requirements.txt
$ python3 setup.py develop
```

#### PyTorch
```
$ pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Specific dependencies 
```
$ pip3 install -r requirements_butlers.txt
```

## Action modeling

Training a GPT-2 medium (345M parameters) model with action modeling on expert demonstrations: 
```
$ python3 src/action_modeling.py configs/action_modeling_config.yaml
```

## Evaluation

Evaluating the resulting model on out-of-distribution task instances: 
```
$ python3 src/alfworld_trainer.py configs/alfworld_trainer_eval_config.yaml
```

## Models

Trained models are hosted on the [Hugging Face hub](https://huggingface.co/vmicheli/lm-butlers-gpt). For instance, set `model_path` to `vmicheli/lm-butlers-gpt` in `alfworld_trainer_eval_config.yaml` to skip the Action modeling step.
