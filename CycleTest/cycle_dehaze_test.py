import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *


class CycleDehazeTest():

    def __init__(self, img_shape=(256, 256, 3), ngf=32, ndf=64, lamdba_cycle=10.0, lambda_id=1.0, lambda_perceptual=1.0):

        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.ngf, self.ndf = ngf, ndf
        self.lambda_cycle, self.lambda_id, self.lambda_perceptual = lamdba_cycle, lambda_id, lambda_perceptual

        self.generator_g_image = self.build_generator()
        self.generator_g_image_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.generator_g_A = self.build_generator()
        self.generator_g_A_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.generator_g_b = self.build_generator()
        self.generator_g_b_optimizer = Adam(2e-4, beta_1=0.5)

        self.discriminator_y = self.build_discriminator()
        self.discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)

        self.vgg16_extract_feature = vgg16_feature_net(img_shape)
        self.vgg16_extract_feature.trainable = False

        self.loss_obj = BinaryCrossentropy(from_logits=True)

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(generator_g_image=self.generator_g_image,
                                   generator_g_A=self.generator_g_A,
                                   generator_g_b=self.generator_g_b,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_image_optimizer=self.generator_g_image_optimizer,
                                   generator_g_A_optimizer=self.generator_g_A_optimizer,
                                   generator_g_b_optimizer=self.generator_g_b_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def build_generator(self):
        return resnet_9blocks(self.img_shape, self.ngf)

    def build_discriminator(self):
        return basic(self.img_shape, self.ndf, use_sigmoid=False)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_cycle * 0.5 * loss

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
    def train_step(self, real_haze, real_clear, real_A, real_b):
        with tf.GradientTape(persistent=True) as tape:

            fake_clear = self.generator_g_image(real_haze, training=True)
            fake_A = self.generator_g_A(real_haze, training=True)
            fake_b = self.generator_g_b(real_haze, training=True)
            recon_fake_haze = fake_clear * fake_A + fake_b
            
            same_clear = self.generator_g_image(real_clear, training=True)

            disc_real_y = self.discriminator_y(real_clear, training=True)
            disc_fake_y = self.discriminator_y(fake_clear, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            id_loss = self.identity_loss(real_clear, same_clear)
            perc_img_loss = self.perceptual_loss(real_clear, fake_clear)
            perc_A_loss = self.perceptual_loss(real_clear, fake_A)
            perc_b_loss = self.perceptual_loss(real_clear, fake_b)
            
            total_gen_g_loss = gen_g_loss + id_loss + perc_img_loss + perc_A_loss + perc_b_loss
            
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_image_gradients = tape.gradient(
            total_gen_g_loss, self.generator_g_image.trainable_variables)
        generator_g_A_gradients = tape.gradient(
            total_gen_g_loss, self.generator_g_A.trainable_variables)
        generator_g_b_gradients = tape.gradient(
            total_gen_g_loss, self.generator_g_b.trainable_variables)
        discriminator_y_gradients = tape.gradient(
            disc_y_loss, self.discriminator_y.trainable_variables)

        self.generator_g_image_optimizer.apply_gradients(
            zip(generator_g_image_gradients, self.generator_g_image.trainable_variables))
        self.generator_g_A_optimizer.apply_gradients(
            zip(generator_g_A_gradients, self.generator_g_A.trainable_variables))
        self.generator_g_b_optimizer.apply_gradients(
            zip(generator_g_b_gradients, self.generator_g_b.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients, self.discriminator_y.trainable_variables))
        return disc_y_loss, total_gen_g_loss

    def train(self, epochs, dataset_haze, dataset_clear, dataset_A, dataset_b, batch_size, loss_interval):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs,))
            for image_haze, image_clear, A, b in tf.data.Dataset.zip((dataset_haze, dataset_clear, dataset_A, dataset_b)).batch(batch_size):
                disc_y_loss, total_gen_g_loss = self.train_step(image_haze, image_clear, A, b)
                if n % loss_interval == 0:
                    print('Loss: Dy: {0} G: {1}'.format(disc_y_loss, total_gen_g_loss))
                n += 1

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} of totoal epoch {} is {}\n'.format(
                epoch + 1,
                epochs,
                datetime.datetime.now() - start))
