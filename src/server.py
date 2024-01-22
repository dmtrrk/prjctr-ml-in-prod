from fastapi import FastAPI, HTTPException, Request
import torch
import os
from utils.common import load_json, load
from utils.app_config import train_config
from utils.model import build_model
from utils.data import tokenize_input, load_metrics

app = FastAPI()

model_path = os.path.join(train_config['output_dir'], load(os.path.join(train_config['output_dir'], train_config['checkpoint_file'])))
model, tokenizer = build_model(model_path)
model.eval()
torch.no_grad()
metrics = load_metrics()
print('Running with metrics:', metrics)

@app.post("/predict")
async def predict(request: Request):
    text = str(await request.body())
    inputs = tokenize_input(text, tokenizer)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    prediction = outputs.logits.squeeze().item()

    return {'target': prediction}

@app.get("/metrics")
async def get_metrics():
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)