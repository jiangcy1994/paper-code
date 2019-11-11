import datetime
from keras.layers import Input, Subtract
from keras.models import Model
import numpy as np

from utils import *

__all__ = ['Derain']

class Derain:
    '''
    Training (Density-aware Deraining network using GT label)
    derain_train_2018.py
    '''
    
    def __init__(self, img_shape=(512, 512, 3), g_filter=64, d_filter=64, lamdba_gan=0.01, lambda_img=1.0):
        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.generator_filter, self.discriminator_filter = g_filter, d_filter
        self.lamdba_gan, self.lambda_img = lamdba_gan, lambda_img

        self.generator = self.build_generator()
        self.generator.name = 'generator'
        
        self.vgg16_feature = VGG16_Feature(img_shape)
        
        self.label_shape = img_shape[:2] + (8,)
        img_input = Input(img_shape, name='img_input')
        label_input = Input(self.label_shape, name='label_input')
        
        residue, x_hat = self.generator([img_input, label_input])
        vgg16_feature = self.vgg16_feature(x_hat)
        
        self.combined = Model(inputs=[img_input, label_input],
                              outputs=[x_hat, vgg16_feature[0], vgg16_feature[1]])
        self.combined.compile('rmsprop', 
                              loss=['mae', 'mae', 'mae'], 
                              loss_weights=[self.lambda_img, 1.8 * self.lambda_img, 1.8 * self.lambda_img])
        
        
    def build_generator(self):
        return Dense_rain(self.img_shape)
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        
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
               
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [loss: %f, l1: %f, features: (%f, %f) ] time: %s " %
                      (epoch, epochs,
                       batch_i, data_loader.n_batches,
                       np.sum(loss), loss[0], loss[1], loss[2], 
                       elapsed_time))
