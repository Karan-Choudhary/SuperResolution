import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
import yaml

vgg19_model = VGG19(weights='imagenet', include_top=False)
feature_extractor = tf.keras.models.Sequential(*[vgg19_model.layers][:18])

def read_params(config_path):
    with open(config_path,'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def generator_loss(disc_generated_output, gen_output, target):
    config = read_params('params.yaml')
    LAMBDA = config['loss']['lambda']
    valid = tf.ones_like(disc_generated_output)
    gan_loss = tf.keras.losses.MSE(valid,disc_generated_output)

    gen_features = feature_extractor(gen_output)
    real_features = feature_extractor(target)
    l1_loss = tf.reduce_mean(tf.abs(gen_features - real_features))

    total_gen_loss = gan_loss + (l1_loss * LAMBDA)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.MSE(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.MSE(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
