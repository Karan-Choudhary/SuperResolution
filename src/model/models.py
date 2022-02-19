import tensorflow as tf
from model.Subpixel.Subpixel_conv2D import SubpixelConv2D
import yaml


def read_params(config_path):
    with open(config_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def get_gen():
    config = read_params('params.yaml')

    global DISC_IMG_HEIGHT, DISC_IMG_WIDTH, DISC_IMG_HEIGHT, CHANNELS, NUM_FILTERS, NUM_KERNELS, STRIDES, PADDING
    IMG_HEIGHT = config['data_load']['LR_img_height']
    IMG_WIDTH = config['data_load']['LR_img_width']
    CHANNELS = config['model']['num_channels']
    NUM_FILTERS = config['model']['num_filters']
    NUM_KERNELS = config['model']['kernel_size']
    NUM_RES_BLOCKS = config['model']['num_res_blocks']
    UPS_FACTOR = config['model']['upsampling_factor']
    STRIDES = config['model']['strides']
    PADDING = config['model']['padding']
    DISC_IMG_HEIGHT = config['data_load']['HR_img_height']
    DISC_IMG_WIDTH = config['data_load']['HR_img_width']

    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    n = tf.keras.layers.Conv2D(NUM_FILTERS, (NUM_KERNELS, NUM_KERNELS), (
        STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init, activation='relu')(nin)
    temp = n

    # B residual blocks
    for i in range(NUM_RES_BLOCKS):
        nn = tf.keras.layers.Conv2D(NUM_FILTERS, (NUM_KERNELS, NUM_KERNELS), (
            STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init)(n)
        nn = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(nn)
        nn = tf.keras.activations.relu(nn)
        nn = tf.keras.layers.Conv2D(NUM_FILTERS, (NUM_KERNELS, NUM_KERNELS), (
            STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init)(nn)
        nn = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(nn)
        n = nn

    n = tf.keras.layers.Conv2D(NUM_FILTERS, (NUM_KERNELS, NUM_KERNELS), (
        STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(n)
    n = tf.keras.layers.Add()([n, temp])

    n = tf.keras.layers.Conv2D(NUM_FILTERS*4, (NUM_KERNELS, NUM_KERNELS),
                               (STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init)(n)
    n = SubpixelConv2D(upsampling_factor=UPS_FACTOR)(n)

    n = tf.keras.layers.Conv2D(NUM_FILTERS*4, (NUM_KERNELS, NUM_KERNELS),
                               (STRIDES, STRIDES), padding=PADDING, kernel_initializer=w_init)(n)
    n = SubpixelConv2D(upsampling_factor=UPS_FACTOR)(n)

    nn = tf.keras.layers.Conv2D(3, (NUM_KERNELS//3, NUM_KERNELS//3), (STRIDES, STRIDES),
                                padding=PADDING, kernel_initializer=w_init, activation='tanh')(n)
    generator = tf.keras.models.Model(inputs=nin, outputs=nn, name='Generator')
    return generator


def get_disc():
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_initializer = tf.random_normal_initializer(1., 0.02)

    nin = tf.keras.layers.Input(
        shape=(DISC_IMG_HEIGHT, DISC_IMG_WIDTH, CHANNELS))
    n = tf.keras.layers.Conv2D(NUM_FILTERS, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2,
                               STRIDES*2), padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU')(nin)

    n = tf.keras.layers.Conv2D(NUM_FILTERS*2, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2, STRIDES*2),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*4, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2, STRIDES*2),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*8, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2, STRIDES*2),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*16, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2, STRIDES*2),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*32, ((NUM_KERNELS*4)//3, (NUM_KERNELS*4)//3), (STRIDES*2, STRIDES*2),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*16, ((NUM_KERNELS)//3, (NUM_KERNELS)//3), (STRIDES, STRIDES),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*8, ((NUM_KERNELS)//3, (NUM_KERNELS)//3), (STRIDES, STRIDES),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    nn = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)

    n = tf.keras.layers.Conv2D(NUM_FILTERS*2, ((NUM_KERNELS)//3, (NUM_KERNELS)//3), (STRIDES, STRIDES),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(nn)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*2, (NUM_KERNELS, NUM_KERNELS), (STRIDES, STRIDES),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Conv2D(NUM_FILTERS*8, (NUM_KERNELS, NUM_KERNELS), (STRIDES, STRIDES),
                               padding=PADDING, kernel_initializer=w_init, activation='LeakyReLU', bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_initializer)(n)
    n = tf.keras.layers.Add()([n, nn])

    n = tf.keras.layers.Flatten()(n)
    n = tf.keras.layers.Dense(
        1, kernel_initializer=w_init, activation='LeakyReLU')(n)
    discriminator = tf.keras.models.Model(
        inputs=nin, outputs=n, name='Discriminator')
    return discriminator
