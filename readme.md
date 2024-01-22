# Projector test
_Source_: https://prjctr.com/course/machine-learning-in-production

_Kaggle_: https://www.kaggle.com/competitions/commonlitreadabilityprize/data

# How to start
```
# train the model
docker-compose run train

# start a server
docker-compose up -d server

# run the test terminal to communicate with the server
docker-compose run client
```

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
