import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
import numpy as np

from utils import *

__all__ = ['AttentiveGanDerainnet']

class AttentiveGanDerainnet:
    
    def __init__(self, img_shape=(128, 128, 3)):
        
        self.img_shape = img_shape
        self.label_shape = self.mask_shape = self.attention_map_shape = img_shape[:-1] + (1,)
        self.cell_state_shape = img_shape[:-1] + (32,)
        
        self.attentive_rnn = build_attentive_rnn(img_shape)
        self.attentive_rnn.name = 'attentive_rnn'
        self.autoencoder = build_autoencoder(img_shape)
        self.autoencoder.name = 'autoencoder'
        self.discriminator = build_discriminative_net(img_shape)
        self.discriminator.name = 'discriminator'
        self.vgg16_feature = VGG16_Feature(img_shape)
        self.vgg16_feature.name = 'vgg16_feature'
        self.vgg16_feature.trainable = False
        
        img_input = Input(self.img_shape, name='img_input')
        zeros_mask = Input(self.mask_shape, name='zeros_mask')
        
        init_attention_map = Input(img_shape[:-1] + (1,), name='init_attention_map')
        init_cell_state = Input(img_shape[:-1] + (32,), name='init_cell_state')
                
        attention_map, lstm_feats, attention_map_1, attention_map_2, attention_map_3, attention_map_4 = self.attentive_rnn([img_input, init_attention_map, init_cell_state])
        
        auto_encoder_input = Concatenate()([attention_map, img_input])
        skip_output_1, skip_output_2, skip_output_3 = self.autoencoder(auto_encoder_input)
        auto_encoder_output = skip_output_3
        
        vgg_features_1, vgg_features_2, vgg_features_3, vgg_features_4, vgg_features_5, vgg_features_6, vgg_features_7, = self.vgg16_feature(auto_encoder_output)
        
        fc_out_o, attention_mask_o, fc_2_o = self.discriminator(auto_encoder_output)
        
        self.combined = Model(
            inputs=[img_input, init_attention_map, init_cell_state, zeros_mask], 
            outputs=[auto_encoder_output, # L_GAN
                     attention_map_1, attention_map_2, attention_map_3, attention_map_4, # L_ATT
                     skip_output_1, skip_output_2, skip_output_3, # L_M
                     vgg_features_1, vgg_features_2, vgg_features_3, vgg_features_4, 
                     vgg_features_5, vgg_features_6, vgg_features_7, # L_P
                     attention_mask_o, zeros_mask, # L_map
                     fc_out_o # L_D - L_map
                    ], 
            name='combined')
                
        self.combined.compile(
            'rmsprop',
            loss=[lambda y_true, y_pred: K.mean(K.log(1.0 - y_pred) , axis=-1), # L_GAN
                  'mse', 'mse', 'mse', 'mse', # L_ATT
                  'mse', 'mse', 'mse', # L_M
                  'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', # L_P
                  'mse', 'mse', # L_map
                  lambda y_true, y_pred: K.mean(-K.log(y_true) - K.log(1.0 - y_pred), axis=-1) # L_D - L_map
                 ],
            loss_weights=[0.01, # L_GAN
                          pow(0.8, 5), pow(0.8, 4), pow(0.8, 3), pow(0.8, 2), # L_ATT
                          0.6, 0.8, 1.0, # L_M
                          1, 1, 1, 1, 1, 1, 1, # L_P
                          0.05, 0.05, # L_map
                          1 # L_D - L_map
                         ])
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=1000):

        start_time = datetime.datetime.now()
        d_loss = 0
        g_loss = 0
        l_gan = 0
        l_att = 0
        l_m = 0
        l_p = 0
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input, mask_input) in enumerate(data_loader.load_batch(batch_size)):
                
                init_attention_map = np.zeros(img_input.shape[:-1] + (1,))
                init_attention_map.fill(0.5)
                init_cell_state = np.zeros(img_input.shape[:-1] + (32,))
                zeros_mask = np.zeros(img_input.shape[:-1] + (1,))
                
                vgg_features_1, vgg_features_2, vgg_features_3, vgg_features_4, vgg_features_5, vgg_features_6, vgg_features_7, = self.vgg16_feature(target_input)

                
                attention_map, _ = self.attentive_rnn.predict(
                    [img_input, init_attention_map, init_cell_state]
                )
                
                fc_out_r, attention_mask_r, _ = self.discriminator.predict(target_input)

                
                loss = self.combined.train_on_batch(
                    [img_input, init_attention_map, init_cell_state, zeros_mask], 
                    [0,
                     mask_input, mask_input, mask_input, mask_input,
                     target_input, target_input, target_input,
                     vgg_features_1, vgg_features_2, vgg_features_3, vgg_features_4, 
                     vgg_features_5, vgg_features_6, vgg_features_7,
                     attention_map, attention_mask_r,
                     fc_out_r]
                )
                
                if batch_i % sample_interval == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    d_loss = np.sum(loss[15:])
                    g_loss = np.sum(loss[:15])
                    l_gan = loss[0]
                    l_att = np.sum(loss[1:5])
                    l_m = np.sum(loss[5:8])
                    l_p = np.sum(loss[8:15])

                    # Plot the progress
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %05f] [G loss: %05f, Lgan: %05f, Latt: %05f, Lm: %05f, Ld: %05f] time: %s " % 
                          (epoch, epochs, 
                           batch_i, data_loader.n_batches,
                           d_loss,
                           g_loss, l_gan, l_att, l_m, l_p,
                           elapsed_time))
        
        print("[D loss: %05f] [G loss: %05f, Lgan: %05f, Latt: %05f, Lm: %05f, Ld: %05f] time: %s " % 
              (d_loss,
               g_loss, l_gan, l_att, l_m, l_p,
               elapsed_time))
                                
    def predict(self, img_input):
        init_attention_map = np.zeros(img_input.shape[:-1] + (1,))
        init_attention_map.fill(0.5)
        init_cell_state = np.zeros(img_input.shape[:-1] + (32,))
        zeros_mask = np.zeros(img_input.shape[:-1] + (1,))
        attention_map, lstm_feats, attention_maps = self.attentive_rnn.predict(
            [img_input, init_attention_map, init_cell_state, zeros_mask]
        )
        np.concat([attention_map, img_input], axis=-1)
        skip_output_1, skip_output_2, skip_output_3 = self.autoencoder.predict(auto_encoder_input)
        return skip_output_3
    
    def save_model(self, dir_name='model'):
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            
        self.attentive_rnn.save_weights(dir_name + '/attentive_rnn.hdf5')
        self.autoencoder.save_weights(dir_name + '/autoencoder.hdf5')
        self.discriminator.save_weights(dir_name + '/discriminator.hdf5')
        self.vgg16_feature.save_weights(dir_name + '/vgg16_feature.hdf5')
        
    def load_model(self, dir_name='model'):
        attentive_rnn = dir_name + '/attentive_rnn.hdf5'
        autoencoder = dir_name + '/autoencoder.hdf5'
        discriminator = dir_name + '/discriminator.hdf5'
        vgg16_feature = dir_name + '/vgg16_feature.hdf5'
        
        if os.path.exists(attentive_rnn) and os.path.isfile(attentive_rnn):
            self.attentive_rnn.load_model(attentive_rnn)
        else:
            print('no attentive_rnn model weights file in {0}'.format(attentive_rnn))
            
        if os.path.exists(autoencoder) and os.path.isfile(autoencoder):
            self.autoencoder.load_model(autoencoder)
        else:
            print('no autoencoder model weights file in {0}'.format(autoencoder))
            
        if os.path.exists(discriminator) and os.path.isfile(discriminator):
            self.discriminator.load_model(discriminator)
        else:
            print('no discriminator model weights file in {0}'.format(discriminator))
            
        if os.path.exists(vgg16_feature) and os.path.isfile(vgg16_feature):
            self.vgg16_feature.load_model(vgg16_feature)
        else:
            print('no vgg16_feature model weights file in {0}'.format(vgg16_feature))
