import tensorflow as tf
import yaml
import time
from model.models import get_gen, get_disc
from model.losses.loss import generator_loss, discriminator_loss
import os


def read_params(config_path):
    with open(config_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


class build():

    def __init__(self):
        self.generator = get_gen()
        self.discriminator = get_disc()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.config = read_params('params.yaml')
        self.SAVED_PATH = self.config['model']['saved_path']

    # def read_params(self,config_path):
    #     with open(config_path,'r') as stream:
    #         try:
    #             params = yaml.safe_load(stream)
    #         except yaml.YAMLError as exc:
    #             print(exc)
    #     return params

    def train(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator(target, training=True)
            disc_generated_output = self.discriminator(
                gen_output, training=True)
            gen_total_loss, gen_loss, gen_loss_l1 = generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))
        return gen_total_loss, disc_loss

    def fit(self, train_ds, epochs):
        for epoch in range(epochs):
            start = time.time()
            for n, (input_image, output_image) in train_ds.enumerate():
                print(".", end="")
                if (n+1) % 100 == 0:
                    print()
                gen_total_loss, disc_loss = self.train(
                    input_image, output_image, epoch)
            print()

            if (epoch+1) % 5 == 0:
                generator.save(os.path.join(SAVED_PATH, "Generator.h5"))
                discriminator.save(os.path.join(
                    SAVED_PATH, "Discriminator.h5"))

            print("==============================================================")
            print("Epoch {} is completed".format(epoch+1))
            print("Time taken = {} sec".format(time.time()-start))
            print("Loss ==> Generator loss = {} , Discriminator loss = {} ".format(
                gen_total_loss, disc_loss))
        return gen_total_loss, disc_loss
