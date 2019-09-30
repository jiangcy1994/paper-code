import datetime
from keras.applications import vgg16
from keras.layers import Input, ZeroPadding2D, Concatenate
from keras.models import Model
import numpy as np
import sys
sys.path.append('../')

from utils.sub_net import *

class CycleDehaze():
    
    def __init__(self, img_shape=(128, 128, 3), g_filter=32, d_filter=64, lamdba_cycle=10.0, lambda_id=1.0, lambda_perceptual=1.0):
        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape

        # Calculate output shape of D (PatchGAN)
#         patch = int(self.img_rows / 2**4)
        patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.generator_filter, self.discriminator_filter = 32, 64

        # Loss weights
        self.lambda_cycle, self.lambda_id, self.lambda_perceptual = lamdba_cycle, lambda_id, lambda_perceptual

        # Build and compile the discriminators
        self.discriminator_A = self.build_discriminator()
        self.discriminator_A.name = 'discriminator_A'
        self.discriminator_A.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        self.discriminator_B = self.build_discriminator()
        self.discriminator_B.name = 'discriminator_B'
        self.discriminator_B.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.generator_A_to_B = self.build_generator()
        self.generator_A_to_B.name = 'generator_A_to_B'
        self.generator_B_to_A = self.build_generator()
        self.generator_B_to_A.name = 'generator_B_to_A'

        # Build the Vgg16 functions
        vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=img_shape)
        vgg16_input = vgg16_model.layers[0].input
        vgg16_2pool_feature = vgg16_model.layers[6].output
        vgg16_5pool_feature = vgg16_model.layers[18].output
        vgg16_5pool_feature_pad = ZeroPadding2D(int((int(vgg16_2pool_feature.shape[1]) - int(vgg16_5pool_feature.shape[1])) / 2))(vgg16_5pool_feature)
        vgg16_all_feature = Concatenate()([vgg16_2pool_feature, vgg16_5pool_feature_pad])

        vgg16_alter = Model(inputs=[vgg16_input], outputs=[vgg16_all_feature])
        self.vgg16_extract_feature = Model(
            inputs=[vgg16_input], 
            outputs=vgg16_all_feature)
        self.vgg16_extract_feature.trainable = False
        
        # Input images from both domains
        img_A = Input(shape=self.img_shape, name='input_A')
        img_B = Input(shape=self.img_shape, name='input_B')

        # Translate images to the other domain
        fake_B = self.generator_A_to_B(img_A)
        fake_A = self.generator_B_to_A(img_B)
        # Translate images back to original domain
        reconstr_A = self.generator_B_to_A(fake_B)
        reconstr_B = self.generator_A_to_B(fake_A)
        # Identity mapping of images
        img_A_id = self.generator_B_to_A(img_A)
        img_B_id = self.generator_A_to_B(img_B)

        # For the combined model we will only train the generators
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.discriminator_A(fake_A)
        valid_B = self.discriminator_B(fake_B)
        
        # Extra vgg16 pool result
        vgg16_feature_A = self.vgg16_extract_feature(reconstr_A)        
        vgg16_feature_B = self.vgg16_extract_feature(reconstr_B)
        
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id,
                                       vgg16_feature_A, vgg16_feature_B])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'mse', 'mse'],
                            loss_weights=[1, 1,
                                          self.lambda_cycle, self.lambda_cycle,
                                          self.lambda_id, self.lambda_id,
                                          self.lambda_perceptual, self.lambda_perceptual
                                         ],
                            optimizer='rmsprop')
    
  
    def build_generator(self):
        return UNET_G(self.img_rows, num_generator_filter=self.generator_filter)
    
    def build_discriminator(self):
        return BASIC_D(self.channels, self.discriminator_filter)
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.generator_A_to_B.predict(imgs_A)
                fake_A = self.generator_B_to_A.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.discriminator_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.discriminator_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.discriminator_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.discriminator_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                        
                # Extra vgg16 pool result
                vgg16_feature_A = self.vgg16_extract_feature.predict(imgs_A)        
                vgg16_feature_B = self.vgg16_extract_feature.predict(imgs_B)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B,
                                                       vgg16_feature_A, vgg16_feature_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f, vgg16_feature: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            np.sum(g_loss[6:8]),
                                                                            elapsed_time))
