import requests
from utils.app_config import client_config

while True:
    text = input("Input excerpt (empty to exit): ")
    if text is None or text == '':
        break
    response = requests.post(f"{client_config['host']}/predict", data=text)
    target = response.json()['target']
    print('Target: ', target)