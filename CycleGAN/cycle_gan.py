import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *

class CycleGAN():
    
    def __init__(self, img_shape=(256, 256, 3), ngf=32, ndf=64, lamdba_cycle=10.0, lambda_id=1.0):

        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.ngf, self.ndf = ngf, ndf
        self.lambda_cycle, self.lambda_id = lamdba_cycle, lambda_id

        self.generator_g = self.build_generator()
        self.generator_g_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.generator_f = self.build_generator()
        self.generator_f_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.discriminator_x = self.build_discriminator()
        self.discriminator_x_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.discriminator_y = self.build_discriminator()
        self.discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)
        
        self.loss_obj = BinaryCrossentropy(from_logits=True)

        self.disc_patch = tuple(self.discriminator_x.output.shape)[1:]
        
        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                   generator_f=self.generator_f,
                                   discriminator_x=self.discriminator_x,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        
    
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
    
    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lamdba_cycle * loss1
    
    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lamdba_cycle * 0.5 * loss
    
    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)
    
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f.trainable_variables))
        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_x.trainable_variables))
        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_y.trainable_variables))
        return disc_x_loss, disc_y_loss, total_gen_g_loss, total_gen_f_loss
    
    def train(self, epochs, datasetX, datasetY, batch_size, loss_interval):
        
        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs,))
            for image_x, image_y in tf.data.Dataset.zip((datasetX, datasetY)).batch(batch_size):
                disc_x_loss, disc_y_loss, total_gen_g_loss, total_gen_f_loss = train_step(image_x, image_y)
                if n % loss_interval == 0:
                    print ('Loss: Dx: {0} Dy: {1} G: {2} F: {3}'.format(disc_x_loss, disc_y_loss, total_gen_g_loss, total_gen_f_loss))
                n += 1
            
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                     ckpt_save_path))


            print ('Time taken for epoch {} of totoal epoch {} is {}\n'.format(
                epoch + 1,
                epochs,
                datetime.datetime.now() - start_time))
        
        ckpt_save_path = self.ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(
            epoch+1,
            ckpt_save_path))
        print ('Time taken is {}\n'.format(datetime.datetime.now() - start_time))
