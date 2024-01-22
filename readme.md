# Projector test
_Source_: https://prjctr.com/course/machine-learning-in-production

_Kaggle_: https://www.kaggle.com/competitions/commonlitreadabilityprize/data

# How to start
```
# train the model with CPU
docker-compose run train

# train the model with CUDA
docker-compose run train_cuda

# start a server
docker-compose up -d server

# run the test terminal to communicate with the server
docker-compose run client
```

# Configuring the train model
Checkout the `config.yaml` for the model configuration

# API definition

1. Getting a prediction based on a text
```
POST /predict
<excerpts>
```

2. Get metrics
```
GET /metrics
```
