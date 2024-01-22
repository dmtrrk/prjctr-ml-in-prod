from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from .app_config import train_config

def build_trainer(model, train_dataset, eval_dataset):
    return RegressionTrainer(
        model=model,
        args=_build_trainer_args(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_regression_metrics
    )

def build_model(model_path=None):
    device = torch.device("cuda" if train_config['use_cuda'] and torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model = AutoModelForSequenceClassification.from_pretrained(train_config['model'] if model_path is None else model_path, num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained(train_config['model'])
    return model, tokenizer

def _build_trainer_args():
    return TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=os.path.join(train_config['output_dir'], 'logs'),
        evaluation_strategy='epoch',
        save_strategy="epoch",
        metric_for_best_model="mse"
    )

def _compute_regression_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    preds = np.squeeze(preds) if len(preds.shape) > 1 else preds

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self._mse_loss(logits.squeeze(-1), labels)
        return (loss, outputs) if return_outputs else loss
    
    def get_steps_number(self):
        print(len(self.train_dataset), train_config['batch_size'], train_config['epochs'])
        return round((len(self.train_dataset) / train_config['batch_size'] + 0.5) * train_config['epochs'])

    def _mse_loss(self, outputs, labels):
        criterion = nn.MSELoss()
        return criterion(outputs, labels)