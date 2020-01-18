import datetime
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import *


class DCPDN():

    def __init__(self, img_shape=(256, 256, 3), lamdba_el2=1.0, lambda_eg=0.5, lambda_ef=0.8, lambda_gan=0.35):

        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.lamdba_el2, self.lambda_eg, self.lambda_ef, self.lambda_gan = lamdba_el2, lambda_eg, lambda_ef, lambda_gan

        self.generator = self.build_generator()
        self.generator_optimizer = Adam(2e-3, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = Adam(2e-3, beta_1=0.5)

        self.vgg16_extract_feature = vgg16_feature_net(img_shape)
        self.vgg16_extract_feature.trainable = False

        self.loss_obj = BinaryCrossentropy(from_logits=True)

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
        return Dehaze(self.img_shape)

    def build_discriminator(self):
        return D(self.img_shape)

    def edge_loss(self, real_trans, generated_trans):
        '''$L_{E}$'''

        # $L_{E,l_2}$
        l2_loss = tf.reduce_mean(tf.square(real_trans - generated_trans))

        # $L_{E,g}$
        def get_gradient_h(x): return tf.abs(x[:, :, :-1] - x[:, :, 1:])
        def get_gradient_v(x): return tf.abs(x[:, :-1] - x[:, 1:])
        lg_loss = tf.reduce_mean(tf.square(get_gradient_h(real_trans) - get_gradient_h(generated_trans))) + \
            tf.reduce_mean(tf.square(get_gradient_v(
                real_trans) - get_gradient_v(generated_trans)))

        # $L_{E,f}$
        real_trans_feature = self.vgg16_extract_feature(real_trans)
        generated_trans_feature = self.vgg16_extract_feature(generated_trans)

        lf_loss = tf.reduce_mean(tf.square(real_trans_feature[0] - generated_trans_feature[0])) + tf.reduce_mean(
            tf.square(real_trans_feature[1] - generated_trans_feature[1]))
        return self.lamdba_el2 * l2_loss + self.lambda_eg * lg_loss + self.lambda_ef * lf_loss

    def atoms_loss(self, real_atoms, generated_atoms):
        '''$L_{A}$'''
        return tf.reduce_mean(tf.square(real_atoms - generated_atoms))

    def dehaze_loss(self, real_img, generated_img):
        '''$L_{D}$'''
        return tf.reduce_mean(tf.square(real_img - generated_img))

    def joint_loss(self, generated_img_label):
        '''$L_{j}$'''
        j_loss = self.loss_obj(tf.ones_like(
            generated_img_label), generated_img_label)
        return self.lambda_gan * j_loss

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    @tf.function
    def train_step(self, image, target, trans, atmos):
        with tf.GradientTape(persistent=True) as tape:

            fake_target, fake_trans, fake_atmos = self.generator(
                image, training=True)

            disc_real = self.discriminator([trans, target], training=True)
            disc_fake = self.discriminator(
                [fake_trans, fake_target], training=True)

            disc_loss = self.discriminator_loss(disc_real, disc_fake)

            gen_loss_E = self.edge_loss(trans, fake_trans)
            gen_loss_A = self.atoms_loss(atmos, fake_atmos)
            gen_loss_D = self.dehaze_loss(target, fake_target)
            gen_loss_j = self.joint_loss(disc_fake)
            total_gen_loss = gen_loss_E + gen_loss_A + gen_loss_D + gen_loss_j

        generator_gradients = tape.gradient(
            total_gen_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))
        return disc_loss, total_gen_loss

    def train(self, epochs, dataset_image, dataset_target, dataset_trans, dataset_atmos, batch_size, loss_interval):

        for epoch in range(epochs):
            start = datetime.datetime.now()

            n = 0
            print('epoch {0}/{1}'.format(epoch + 1, epochs))
            for image, target, trans, atmos in tf.data.Dataset.zip((dataset_image, dataset_target, dataset_trans, dataset_atmos)).batch(batch_size):
                disc_loss, total_gen_loss = self.train_step(
                    image, target, trans, atmos)
                if n % loss_interval == 0:
                    print('Batch:{0} | Loss: | D: {1} | G: {2}'.format(
                        n, disc_loss, total_gen_loss))
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
