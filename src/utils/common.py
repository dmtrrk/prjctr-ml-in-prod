import json

def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save(filename, text):
    with open(filename, 'w') as f:
        f.write(text)

def load(filename):
    with open(filename, 'r') as f:
        return f.read()