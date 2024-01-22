import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_config = config["train"]
client_config = config["client"]
