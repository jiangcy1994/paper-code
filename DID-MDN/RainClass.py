import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from utils import *

__all__ = ['RainClass']


class RainClass:
    '''
    Density-estimation Training (rain-density classifier)
    train_rain_class.py
    '''

    def __init__(self, img_shape=(512, 512, 3), num_classes=3):
        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.label_shape = img_shape[:2] + (4,)
        self.num_classes = num_classes

        self.generator = self.build_generator()
        self.generator_optimizer = Adam(1e-3, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = Adam(1e-3, beta_1=0.5)

        self.loss_obj = CategoricalCrossentropy(from_logits=True)

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

    def build_discriminator(self):
        return VGG19ca(self.img_shape)

    def build_generator(self):
        return Dense_rain_residual(self.img_shape)

    def euclidean_residual_loss(self, real_residual, fake_residual):
        return tf.reduce_mean(tf.square(real_residual - fake_residual))

    def crossentropy_loss(self, real_label, fake_label):
        real_label_class = tf.keras.utils.to_categorical(real_label, self.num_classes=3)
        return self.loss_obj(real_label_class, fake_label)

    @tf.function
    def train_step(self, image, label, target):
        with tf.GradientTape(persistent=True) as tape:

            zero_label = tf.zeros(self.label_shape)
            output = self.generator([image, zero_label], training=True)
            fake_residue = image - output
            real_residue = image - target
            fake_label = self.discriminator(residue, training=True)

            l_er = self.euclidean_residual_loss(real_residual, fake_residual)
            l_c = self.crossentropy_loss(label, fake_label)

            total_loss = l_er + l_c

        generator_gradients = tape.gradient(
            total_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(
            total_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))
        return total_loss

    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs))
            for image, label, target in tf.data.Dataset.zip((dataset_image, dataset_label, dataset_target)).batch(batch_size):
                total_loss = self.train_step(image, label, target)
                if n % loss_interval == 0:
                    print('Batch:{0} | Loss: {1}'.format(
                        n, total_loss))
                n += 1

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
