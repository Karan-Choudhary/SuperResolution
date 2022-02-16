import tensorflow as tf
import yaml


def read_params(config_path):
    with open(config_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


config = read_params('params.yaml')
HR_img_width = config['data_load']['HR_img_width']
HR_img_height = config['data_load']['HR_img_height']
LR_image_width = config['data_load']['LR_img_width']
LR_image_height = config['data_load']['LR_img_height']


def load(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    return image


def resize(image, height, width):
    image = tf.image.resize(
        image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def normalize(image):
    image = (image/127.5) - 1
    return image


def random_crop(image):
    image = tf.image.random_crop(image, [HR_img_height, HR_img_width, 3])
    return image


@tf.function
def random_jitter(image):
    image = resize(image, HR_img_height+6, HR_img_width+6)
    image = random_crop(image)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image


def load_image_train(image_file):
    image = load(image)
    HR_image = random_jitter(image)
    LR_image = resize(HR_image, LR_image_height, LR_image_width)
    HR_image = normalize(HR_image)
    LR_image = normalize(LR_image)
    return LR_image, HR_image


def load_image_test(image_file):
    image = load(image)
    image = resize(image, HR_img_height, HR_img_width)
    image = normalize(image)
    return image
