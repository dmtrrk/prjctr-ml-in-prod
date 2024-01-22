import os
from utils.model import build_trainer, build_model
from utils.data import load_data
from utils.common import save_json, save
from utils.app_config import train_config

model, tokenizer = build_model()
train_data, eval_data, test_data = load_data(tokenizer)
trainer = build_trainer(model, train_data, eval_data)

trainer.train()

metrics = trainer.evaluate(test_data)
save_json(os.path.join(train_config['output_dir'], 'metrics.json'), metrics)

save(os.path.join(train_config['output_dir'], train_config['checkpoint_file']), f"checkpoint-{trainer.get_steps_number()}")
