from get_data import get_data
from model.build import build
import os
import yaml
import tensorflow as tf
import argparse
from read_params import read_params
import json


def train_model(config_path):
    config = read_params(config_path)
    EPOCHS = config['training']['epochs']
    train_ds, _ = get_data(config_path)
    model = build()
    generator_loss, discriminator_loss = model.fit(train_ds, epochs=100)
    # generator_loss, discriminator_loss = 2,3

    logs_file = config['reports']['logs']
    params_file = config['reports']['params']

    with open(logs_file,'w') as f:
        score = {
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss,
        }
        json.dump(score, f, indent=4)
    
    with open(params_file, 'w') as f:
        params = {
            'epochs': EPOCHS,
            'lambda': config['loss']['lambda'],
            'batch_size': config['data_load']['batch_size'],
            'upsmapling_factor': config['model']['upsampling_factor']
        }
        json.dump(params, f, indent=4)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml', help="params file")
    parsed_args = args.parse_args()
    train_model(config_path=parsed_args.config)