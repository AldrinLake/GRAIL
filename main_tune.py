import json
import argparse
from trainer import train
import os


def main():
    config_json = os.getenv("CONFIG_JSON")
    if config_json:
        config = json.loads(config_json)
    else:
        raise ValueError("CONFIG_JSON environment variable is missing!")
    args = config
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

if __name__ == '__main__':
    main()
