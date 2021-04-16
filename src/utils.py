import random

import numpy as np
import torch

from transformers import DataCollatorForLanguageModeling, BatchEncoding
from torch.utils.data import Dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorForActionModeling(DataCollatorForLanguageModeling):
    # Ensures that the LM loss is only computed over actions
    def __post_init__(self):
        self.state_id = self.tokenizer('STATE')['input_ids'][0]
        self.end_modality_id = self.tokenizer(']')['input_ids'][0]
        self.action_id = self.tokenizer('ACTION')['input_ids'][0]

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        for example_index, example in enumerate(batch['input_ids'].tolist()):
            keep = False
            for tok_index, tok_id in enumerate(example):
                if tok_id == self.state_id:
                    keep = False
                if not keep or tok_id == self.end_modality_id:
                    labels[example_index, tok_index] = -100
                if tok_id == self.action_id:
                    keep = True

        batch["labels"] = labels

        return batch


class DemonstrationsDataset(Dataset):
    def __init__(self, demonstrations):
        self.demonstrations = demonstrations

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, i):
        return self.demonstrations[i]
