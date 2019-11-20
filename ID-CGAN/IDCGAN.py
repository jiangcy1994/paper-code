import datetime
from keras import backend as K
from keras.layers import Concatenate, Input, Lambda
from keras.models import Model
import numpy as np

from utils import *

__all__ = ['IDCGAN']

class IDCGAN:
    
    def __init__(self, img_shape=(256, 256, 3), lambda_0=1, lambda_1=0.35):
        self.img_shape = img_shape
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        
        img_input = Input(self.img_shape)
        target_input = Input(self.img_shape)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.name = 'discriminator'
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer='rmsprop')
        self.disc_patch = self.discriminator.output_shape[1:]
        
        self.generator = self.build_generator()
        self.generator.name = 'generator'
        
        clear_hat = self.generator(img_input)
        fake_output = self.discriminator([img_input, clear_hat])
        real_output = self.discriminator([img_input, target_input])
        
        self.combined = Model(
            inputs=[img_input, target_input],
            outputs=[
                real_output, fake_output, # L_D
                clear_hat, # L_GAN
                clear_hat, # L_Perceptual
                clear_hat, # L_Eucledean
            ])
        self.combined.compile(
            loss=[
                'binary_crossentropy', 'binary_crossentropy',
                'binary_crossentropy',
                'mean_absolute_percentage_error', 
                'mse'
            ],
            loss_weights=[
                0.5, 0.5,
                1,
                self.lambda_0,
                self.lambda_1
            ],
            optimizer='rmsprop')
    
    def build_generator(self):
        return G(self.img_shape)
    
    def build_discriminator(self):
        input0 = Input(self.img_shape)
        input1 = Input(self.img_shape)
        inputs = Concatenate()([input0, input1])
        D_model = D()
        return Model(inputs=[input0, input1], outputs=[D_model(inputs)])
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        d_loss = 0
        g_loss = []

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input) in enumerate(data_loader.load_batch(batch_size)):

                # Train the generators
                losses = self.combined.train_on_batch(
                    [img_input, target_input],
                    [valid, fake,
                     target_input,
                     target_input,
                     target_input
                    ])
                
                if batch_i % sample_interval == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    d_loss = np.mean(losses[:2])
                    g_loss = loss[2:]

                    # Plot the progress
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %05f, Lg: %05f, Lp: %05f, Le: %05f] time: %s " %
                           (epoch, epochs,
                            batch_i, data_loader.n_batches,
                            d_loss,
                            np.sum(g_loss),
                            g_loss[0],
                            g_loss[1],
                            g_loss[2],
                            elapsed_time))
        
        elapsed_time = datetime.datetime.now() - start_time
        d_loss = np.mean(losses[:2])
        g_loss = loss[2:]
        print ("[D loss: %f] [G loss: %05f, Lg: %05f, Lp: %05f, Le: %05f] time: %s " %
               (d_loss,
                np.sum(g_loss),
                g_loss[0],
                g_loss[1],
                g_loss[2],
                elapsed_time))

    def predict(self, img_input):
        img_output = self.generator.predict([img_input])
        return img_output
    
    def save_model(self, dir_name='model'):
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            
        self.discriminator.save_weights(dir_name + '/idc_gan_discriminator.hdf5')
        self.generator.save_weights(dir_name + '/idc_gan_generator.hdf5')
        
    def load_model(self, dir_name='model'):
        discriminator_name = dir_name + '/idc_gan_discriminator.hdf5'
        generator_name = dir_name + '/idc_gan_generator.hdf5'
        
        if os.path.exists(discriminator_name) and os.path.isfile(discriminator_name):
            self.discriminator.load_model(discriminator_name)
        else:
            print('no discriminator model weights file in {0}'.format(discriminator_name))

        if os.path.exists(generator_name) and os.path.isfile(generator_name):
            self.generator.load_model(generator_name)
        else:
            print('no discriminator model weights file in {0}'.format(generator_name))
    