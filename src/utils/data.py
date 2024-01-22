from datasets import Dataset
from .app_config import train_config
from .common import load_json
import os

def load_data(tokenizer):
    dataset = Dataset.from_csv(train_config['train_csv'])
    remove_columns = [col for col in dataset.column_names if col not in ['excerpt', 'target']]
    dataset = dataset.map(lambda batch: _tokenizer(batch, tokenizer), batched=True, remove_columns=remove_columns)

    train_split = dataset.train_test_split(test_size=train_config['eval_size'])
    test_split = train_split['test'].train_test_split(test_size=0.5)

    return train_split['train'], test_split['train'], test_split['test']

def _tokenizer(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["excerpt"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = examples["target"]
    return tokenized_inputs

def tokenize_input(text, tokenizer):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return inputs

def load_metrics():
    return load_json(os.path.join(train_config['output_dir'], 'metrics.json'))