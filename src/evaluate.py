from get_data import get_data
from read_params import read_params
import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from display.showOutput import generate_images

def evaluate_model(config_path):
    config = read_params(config_path)
    SAMPLES = config['test']['num_samples']
    MODEL_DIR = config['model']['saved_path']
    RESULTS = config['generated_image_output_path']
    
    test_ds, _ = get_data(config_path)
    
    model = load_model(os.path.join(MODEL_DIR,'Generator.h5'))

    for (input_image,target_image) in test_ds.take(SAMPLES):
        generate_images(model,input_image,target_image,RESULTS)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml', help="params file")
    parsed_args = args.parse_args()
    evaluate_model(config_path = args.config_path)