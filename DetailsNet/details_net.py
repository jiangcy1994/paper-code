import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *


class DetailsNet:

    def __init__(self, img_shape=(512, 512, 3)):
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape

        self.generator = self.build_generator()
        self.generator_optimizer = Adam(2e-4, beta_1=0.5)

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   generator_optimizer=self.generator_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def cal_detail(self, image):
        base = guided_filter(image, image, 15, 1, nhwc=True)
        return image - base

    def build_generator(self):
        return inference(self.img_shape)

    def image_loss(self, real_target, fake_target):
        return tf.reduce_mean(tf.square(real_target - fake_target))

    @tf.function
    def train_step(self, image, target):
        with tf.GradientTape(persistent=True) as tape:

            detail = self.cal_detail(image)
            fake_target = self.generator([image, detail], training=True)

            loss = self.image_loss(target, fake_target)

        generator_gradients = tape.gradient(
            loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        return loss

    def train(self, epochs, dataset_image, dataset_target, batch_size, loss_interval):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs,))
            for image, target in tf.data.Dataset.zip((dataset_image, dataset_target)).batch(batch_size):
                loss = self.train_step(image, target)
                if n % loss_interval == 0:
                    print('Batch:{0} | Loss: {1}'.format(n, loss))
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
