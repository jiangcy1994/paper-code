import datetime
import gc
from keras import backend as K
from keras.layers import Concatenate, Input, Lambda
from keras.models import Model
import numpy as np

from utils import *

__all__ = ['DCPDN']

class DCPDN:
    
    def __init__(self, img_shape=(256, 256, 3), lambda_img=1, lambda_gan=0.35):
        self.img_shape = img_shape
        self.lambda_img = lambda_img
        self.lambda_gan = lambda_gan
        self.vgg16_feature = VGG16_Feature(img_shape)
        self.vgg16_feature.name = 'vgg16_feature'
        
        label_real = np.ones(self.img_shape)
        label_fake = np.zeros(self.img_shape) 
        
        img_input = Input(self.img_shape)
        target_input = Input(self.img_shape)
        trans_input = Input(self.img_shape)
        ato_input = Input(self.img_shape)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.name = 'discriminator'
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer='rmsprop')
        self.disc_patch = self.discriminator.output_shape[1:]
        
        self.generator = self.build_generator()
        self.generator.name = 'generator'
        
        x_hat, trans_hat, atp_hat, dehaze21 = self.generator(img_input)
        fake_output = self.discriminator([trans_hat, x_hat])
        
        self.get_gradient_h = Lambda(function=lambda x:K.abs(x[:, :, :, :-1] - x[:, :, :, 1:]), name='get_gradient_h')
        self.get_gradient_v = Lambda(function=lambda x:K.abs(x[:, :, :-1, :] - x[:, :, 1:, :]), name='get_gradient_v')
        
        gradient_h_hat = self.get_gradient_h(trans_hat)
        gradient_v_hat = self.get_gradient_v(trans_hat)
        features_content0 ,features_content1 = self.vgg16_feature(trans_hat)
        
        self.combined = Model(inputs=[img_input, target_input, trans_input, ato_input],
                              outputs=[x_hat, # L_L1 for overall
                                       trans_hat, # L1 for transamission map
                                       gradient_h_hat, gradient_v_hat, # gradient loss for transamission map
                                       features_content0, features_content1, # feature loss for transmission map
                                       atp_hat, # L1 for atop-map
                                       fake_output # gan_loss for the joint discriminator
                                      ])
        self.combined.compile(loss=['mae', 
                                    'mae',
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'mae',
                                    'binary_crossentropy'
                                   ],
                              loss_weights=[self.lambda_img,
                                            self.lambda_img,
                                            2 * self.lambda_img, 2 * self.lambda_img,
                                            0.8 * self.lambda_img, 0.8 * self.lambda_img,
                                            self.lambda_img,
                                            self.lambda_gan
                                           ],
                              optimizer='rmsprop'
                             )
    
    def build_generator(self):
        return Dehaze()
    
    def build_discriminator(self):
        input0 = Input((512, 512, 3))
        input1 = Input((512, 512, 3))
        inp = Concatenate()([input0, input1])
        D_model = D()
        model = Model(inputs=[input0, input1], outputs=[D_model(inp)])
        return model
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input, trans_input, ato_input) in enumerate(data_loader.load_batch(batch_size)):

                x_hat, trans_hat, atp_hat, dehaze21 = self.generator.predict(img_input)
                # ----------------------
                #  Train Discriminators
                # ----------------------
#                 self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(
                    [trans_input, target_input], 
                    valid
                )
                d_loss_fake = self.discriminator.train_on_batch(
                    [trans_hat, x_hat],
                    fake
                )
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ------------------
                #  Train Generators
                # ------------------
#                 self.discriminator.trainable = False
#                 gradient_h = self.get_gradient_h(trans_input_tensor)
#                 gradient_v = self.get_gradient_v(trans_input_tensor)
                gradient_h = self.get_gradient_h.call(trans_input)
                gradient_v = self.get_gradient_v.call(trans_input)
                features_content0 ,features_content1 = self.vgg16_feature.predict(trans_input)
                
                # Train the generators
                g_loss = self.combined.train_on_batch([img_input, target_input, trans_input, ato_input],
                                                      [img_input,
                                                       trans_input,
                                                       gradient_h, gradient_v,
                                                       features_content0, features_content1,
                                                       ato_input,
                                                       valid
                                                      ])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %05f, Lt: %05f, La: %05f, Ld: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, data_loader.n_batches,
                                                                            d_loss,
                                                                            g_loss[7],
                                                                            np.sum(g_loss[1:6]),
                                                                            g_loss[6],
                                                                            g_loss[0],
                                                                            elapsed_time))
