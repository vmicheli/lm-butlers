import sys

from yaml import safe_load

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

from utils import DataCollatorForActionModeling

config = safe_load(open(sys.argv[1]))

# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config['model_path'])

# Create dataset

train_data_files = [config['data_path']]

dataset = load_dataset('text', data_files=train_data_files, split='train')
dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, max_length=config['max_seq_length']), batched=True, batch_size=100000)
dataset.remove_columns_('attention_mask')
dataset.set_format(type='torch', columns=['input_ids'])

# Create Trainer

data_collator = DataCollatorForActionModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(**config['training_args'])

trainer = Trainer(model=model, data_collator=data_collator, train_dataset=dataset, args=training_args)

# Train
trainer.train()
