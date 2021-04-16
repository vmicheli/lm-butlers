# Language Models are Few-Shot Butlers

This repository contains the code to reproduce the experiments in Language Models are Few-Shot Butlers.

## Installation

pip install -r requirements.txt

## Action modeling

Training a GPT-2 medium (345M parameters) model with action modeling on expert demonstrations: python src/action_modeling.py configs/action_modeling_config.yaml

## Evaluation

Evaluating the resulting model on out-of-distribution task instances: python src/alfworld_trainer.py configs/alfworld_trainer_eval_config.yaml
