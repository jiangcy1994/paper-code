import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *

__all__ = ['Derain']


class Derain:
    '''
    Training (Density-aware Deraining network using GT label)
    derain_train_2018.py
    '''

    def __init__(self, img_shape=(512, 512, 3), lambda_F=1):

        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.label_shape = img_shape[:2] + (8,)
        self.lambda_F = lambda_F

        self.generator = self.build_generator()
        self.generator_optimizer = Adam(1e-3, beta_1=0.5)

        self.vgg16_extract_feature = vgg16_feature_net(img_shape)
        self.vgg16_extract_feature.trainable = False

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   generator_optimizer=self.generator_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def build_generator(self):
        return Dense_rain(self.img_shape)

    def euclidean_derain_loss(self, real_target, fake_target):
        return tf.reduce_mean(tf.square(real_target - fake_target))

    def feature_based_loss(self, real_target, fake_target):
        real_trans_feature = self.vgg16_extract_feature(real_target)
        generated_trans_feature = self.vgg16_extract_feature(fake_target)
        loss = tf.reduce_mean(tf.square(real_trans_feature[0] - generated_trans_feature[0])) + tf.reduce_mean(
            tf.square(real_trans_feature[1] - generated_trans_feature[1]))
        return self.lambda_F * loss

    @tf.function
    def train_step(self, image, label, target):
        with tf.GradientTape(persistent=True) as tape:

            fake_residue, fake_target = self.generator(
                [image, label], training=True)
            real_residue = tf.subtract(target, label)

            l_ed = self.euclidean_derain_loss(target, fake_target)
            l_f = self.feature_based_loss(target, fake_target)

            total_gen_loss = l_er + l_ed + l_f

        generator_gradients = tape.gradient(
            total_gen_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        return total_gen_loss

    def train(self, epochs, dataset_image, dataset_label, dataset_target,  batch_size, loss_interval):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs))
            for image, label, target in tf.data.Dataset.zip((dataset_image, dataset_label, dataset_target)).batch(batch_size):
                total_gen_loss = self.train_step(image, label, target)
                if n % loss_interval == 0:
                    print('Batch:{0} | Loss: | G: {1}'.format(
                        n, total_gen_loss))
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
