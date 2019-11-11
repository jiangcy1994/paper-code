import datetime
from keras.layers import Input, Subtract
from keras.models import Model
import numpy as np

from utils import *

__all__ = ['RainClass']

class RainClass:
    '''
    Density-estimation Training (rain-density classifier)
    train_rain_class.py
    '''
    
    def __init__(self, img_shape=(512, 512, 3)):
        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = img_shape
        self.label_shape = img_shape[:2] + (4,)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.name = 'discriminator'
        self.generator = self.build_generator()
        self.generator.name = 'generator'
        
        img_input = Input(img_shape, name='img_input')
        zero_label = Input(self.label_shape, name='zero_label')
        
        output = self.generator([img_input, zero_label])
        residue = Subtract()([img_input, output])
        label = self.discriminator(residue)
        
        self.combined = Model(inputs=[img_input, zero_label],
                              outputs=[label])
        self.combined.compile('rmsprop', 
                              loss=['categorical_crossentropy'])
    
    def build_discriminator(self):
        return VGG19ca(self.img_shape)
        
    def build_generator(self):
        return Dense_rain_residual(self.img_shape)
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input, label) in enumerate(data_loader.load_batch(batch_size)):
                zero_label = np.zeros((label.shape[0],) + self.label_shape)
                label_input = np.zeros((label.shape[0],) + self.label_shape)
                for i, one_label in enumerate(label_input):
                    one_label.fill(label[i])
                
                loss = self.combined.train_on_batch(
                    [img_input, zero_label],
                    [label_input]
                )
               
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] time: %s " %
                      (epoch, epochs,
                       batch_i, data_loader.n_batches,
                       np.sum(loss), 
                       elapsed_time))