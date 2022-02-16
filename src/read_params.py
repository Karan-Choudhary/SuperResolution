import yaml
import argparse

def read_params(config_path):
    with open(config_path,'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml', help='params file')
    parsed_args = args.parse_args()
    config = read_params(config_path = parsed_args.config)