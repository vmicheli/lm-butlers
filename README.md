# Language Models are Few-Shot Butlers

This repository contains the code to reproduce the experiments in [Language Models are Few-Shot Butlers](https://arxiv.org/abs/2104.07972).

## Installation

`pip install -r requirements.txt`

## Action modeling

Training a GPT-2 medium (345M parameters) model with action modeling on expert demonstrations: `python src/action_modeling.py configs/action_modeling_config.yaml`

## Evaluation

Evaluating the resulting model on out-of-distribution task instances: `python src/alfworld_trainer.py configs/alfworld_trainer_eval_config.yaml`

## Models

Trained models are hosted on the [Hugging Face hub](https://huggingface.co/vmicheli/lm-butlers-gpt). For instance, set `model_path` to `vmicheli/lm-butlers-gpt` in `alfworld_trainer_eval_config.yaml` to skip the Action modeling step.
