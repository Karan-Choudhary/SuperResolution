import yaml
import argparse
import tensorflow as tf
from read_params import read_params
from utils.preprocessing import load, resize, normalize, random_jitter


def load_image_train(image_file):
    HR_image = load(image_file)
    HR_image = random_jitter(HR_image)
    LR_image = resize(HR_image, LR_img_height, LR_img_width)
    HR_image = normalize(HR_image)
    LR_image = normalize(LR_image)
    return LR_image, HR_image


def load_image_test_HR(image_file):
    image = load(image_file)
    image = resize(image, HR_img_height, HR_img_width)
    image = normalize(image)
    return image


def load_image_test_LR(image_file):
    image = load(image_file)
    image = resize(image, LR_img_height, LR_img_width)
    image = normalize(image)
    return image


def get_data(config_path):

    config = read_params(config_path)
    global train_path, test_path_LR, test_path_HR, BATCH_SIZE, BUFFER_SIZE, LR_img_width, LR_img_height, HR_img_width, HR_img_height
    train_path = config['data_path']['train']
    test_path_LR = config['data_path']['test_LR']
    test_path_HR = config['data_path']['test_HR']
    BATCH_SIZE = config['data_load']['batch_size']
    BUFFER_SIZE = config['data_load']['buffer_size']
    LR_img_width = config['data_load']['LR_img_width']
    LR_img_height = config['data_load']['LR_img_height']
    HR_img_width = config['data_load']['HR_img_width']
    HR_img_height = config['data_load']['HR_img_height']

    # Training data
    train_dataset = tf.data.Dataset.list_files(train_path+'\\*.png')
    train_dataset = train_dataset.map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Test data
    test_dataset_LR = tf.data.Dataset.list_files(
        test_path_LR+'\\*.png', shuffle=False)
    test_dataset_HR = tf.data.Dataset.list_files(
        test_path_HR+'\\*.png', shuffle=False)
    test_dataset_LR = test_dataset_LR.map(
        load_image_test_LR, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset_HR = test_dataset_HR.map(
        load_image_test_HR, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.zip((test_dataset_LR, test_dataset_HR))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml', help="params file")
    parsed_args = args.parse_args()
    train_dataset, test_dataset = get_data(config_path=parsed_args.config)
    print(train_dataset, test_dataset)
