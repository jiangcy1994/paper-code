import datetime
from tensorflow.keras.layers import Input, Subtract
from tensorflow.keras.models import Model
import numpy as np
import os
from utils import *

__all__ = ['Derain']

class Derain:
    '''
    Training (Density-aware Deraining network using GT label)
    derain_train_2018.py
    '''
    
    def __init__(self, img_shape=(512, 512, 3), lambda_F=1.8):
        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.label_shape = img_shape[:2] + (8,)
        self.lambda_F = lambda_F

        self.generator = self.build_generator()
        self.generator.name = 'generator'
        self.vgg16_feature = vgg16_feature_net(img_shape)
        
        img_input = Input(img_shape, name='img_input')
        label_input = Input(self.label_shape, name='label_input')
        
        residue, x_hat = self.generator([img_input, label_input])
        vgg16_feature = self.vgg16_feature(x_hat)
        
        self.combined = Model(inputs=[
                                  # input image
                                  img_input,
                                  # input label
                                  label_input],
                              outputs=[
                                  # L_{E,r} not found
                                  # L_{E,d}
                                  x_hat, 
                                  # L_F
                                  # actually only vgg16_feature[0] is written in the artical
                                  vgg16_feature[0], vgg16_feature[1]
                              ])
        self.combined.compile('rmsprop',
                              loss=['mse', 'mse', 'mse'], 
                              loss_weights=[1, self.lambda_F, self.lambda_F])
        
        
    def build_generator(self):
        return Dense_rain(self.img_shape)
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        total_loss = 0
        l2_loss = 0
        feature_loss0 = 0
        feature_loss1 = 0
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input, label) in enumerate(data_loader.load_batch(batch_size)):
                vgg16_feature = self.vgg16_feature.predict(target_input)
                label_input = np.zeros((label.shape[0],) + self.label_shape)
                for i, one_label in enumerate(label_input):
                    one_label.fill(label[i])
                
                loss = self.combined.train_on_batch(
                    [img_input, label_input],
                    [target_input, vgg16_feature[0], vgg16_feature[1]]
                )
               
                # Plot the progress
                if batch_i % sample_interval == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    total_loss = np.sum(loss)
                    l2_loss = loss[0]
                    feature_loss0 = loss[1]
                    feature_loss1 = loss[2]
                    print("[Epoch %d/%d] [Batch %d/%d] [loss: %f, l2: %f, features: (%f, %f)] time: %s " %
                          (epoch, epochs,
                           batch_i, data_loader.n_batches * 3,
                           total_loss, l2_loss, feature_loss0, feature_loss1, 
                           elapsed_time))
        
        elapsed_time = datetime.datetime.now() - start_time
        # Print Result After Train
        print("[loss: %f, l2: %f, features: (%f, %f)] time: %s " %
              (total_loss, l2_loss, feature_loss0, feature_loss1, 
               elapsed_time))
    
    def predict(self, img_input, label):
        label_input = np.zeros((label.shape[0],) + self.label_shape)
        for i, one_label in enumerate(label_input):
            one_label.fill(label[i])
        result = self.generator.predict([img_input, label_input])
        return result
    
    def save_model(self, dir_name='model'):
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            
        self.generator.save_weights(dir_name + '/derain.hdf5')
        
    def load_model(self, dir_name='model'):
        file_name = dir_name + '/derain.hdf5'
        if os.path.exists(file_name) and os.path.isfile(file_name):
            self.generator.load_model(file_name)
        else:
            print('no model weights file in {0}'.format(file_name))
