import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *


class Pix2pix():
    '''Direction X to Y'''

    def __init__(self, img_shape=(256, 256, 3), ngf=64, ndf=64, lamdba_l1=100.0):

        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.ngf, self.ndf = ngf, ndf
        self.lamdba_l1 = lamdba_l1

        self.generator = self.build_generator()
        self.generator_optimizer = Adam(2e-4, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = Adam(2e-4, beta_1=0.5)

        self.loss_obj = BinaryCrossentropy(from_logits=True)

        self.disc_patch = tuple(self.discriminator_x.output.shape)[1:]

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   discriminator=self.discriminator,
                                   generator_optimizer=self.generator_optimizer,
                                   discriminator_optimizer=self.discriminator_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def build_generator(self):
        return unet(self.img_shape, self.ngf)

    def build_discriminator(self):
        return basic(self.img_shape, self.ndf, use_sigmoid=False)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def l1_loss(self, real, generated):
        return self.lamdba_l1 * tf.reduce_mean(tf.abs(real - generated))

    def perceptual_loss(self, real_image, generated_image):
        real_features = self.vgg16_extract_feature(real_image)
        generated_features = self.vgg16_extract_feature(generated_image)
        loss_1 = tf.reduce_mean(tf.abs(
            real_features[0] -
            generated_features[0]))
        loss_2 = tf.reduce_mean(tf.abs(
            real_features[1] -
            generated_features[1]))
        loss = loss_1 + loss_2
        return self.lambda_perceptual * loss

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:

            fake_y = self.generator(real_x, training=True)
            disc_real_y = self.discriminator(real_y, training=True)
            disc_fake_y = self.discriminator(fake_y, training=True)

            gen_loss = self.generator_loss(disc_fake_y)
            l1_loss = self.l1_loss(real_y, fake_y)

            total_gen_loss = gen_g_loss + l1_loss

            disc_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_gradients = tape.gradient(
            total_gen_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))
        return disc_loss, total_gen_loss

    def train(self, epochs, datasetX, datasetY, batch_size, loss_interval):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs,))
            for image_x, image_y in tf.data.Dataset.zip((datasetX, datasetY)).batch(batch_size):
                disc_loss, total_gen_loss = self.train_step(image_x, image_y)
                if n % loss_interval == 0:
                    print('Loss: D: {0} G: {1}'.format(
                        disc_loss, total_gen_loss))
                n += 1

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} of totoal epoch {} is {}\n'.format(
                epoch + 1,
                epochs,
                datetime.datetime.now() - start))

        ckpt_save_path = self.ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch+1,
            ckpt_save_path))
        print('Time taken is {}\n'.format(datetime.datetime.now() - start))
