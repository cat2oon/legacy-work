import json


def load_json(json_path):
    try:
        with open(json_path) as json_data:
            data = json.load(json_data)
        return data
    except FileNotFoundError:
        return None


def write_json(json_path, json_data):
    with open(json_path, 'w+') as f:
        json.dump(json_data, f)
