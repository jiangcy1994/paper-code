import datetime
from keras.layers import Input, Subtract
from keras.models import Model
from keras.utils import to_categorical
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
                              outputs=[residue, label])
        self.combined.compile('rmsprop', 
                              loss=['mse', 'categorical_crossentropy'])
    
    def build_discriminator(self):
        return VGG19ca(self.img_shape)
        
    def build_generator(self):
        return Dense_rain_residual(self.img_shape)
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        total_loss = 0
        l_Er = 0
        l_C = 0
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input, label) in enumerate(data_loader.load_batch(batch_size)):
                zero_label = np.zeros((label.shape[0],) + self.label_shape)
                label_input = to_categorical(label, 4)
                residue = img_input - target_input
                
                loss = self.combined.train_on_batch(
                    [img_input, zero_label],
                    [residue, label_input]
                )
               
                # Plot the progress
                if batch_i % sample_interval == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    total_loss = np.sum(loss)
                    l_Er = loss[0]
                    l_C = loss[1]
                    print("[Epoch %d/%d] [Batch %d/%d] [loss: %f, l_Er: %f, l_C:%f] time: %s " %
                          (epoch, epochs,
                           batch_i, data_loader.n_batches * 3,
                           total_loss, l_Er, l_C,
                           elapsed_time))
        
        elapsed_time = datetime.datetime.now() - start_time
        # Print Result After Train
        print("[loss: %f, l_Er: %f, l_C:%f] time: %s " %
              (total_loss, l_Er, l_C, elapsed_time))
            
    def predict(self, img_input):
        zero_label = np.zeros((label.shape[0],) + self.label_shape)
        result = self.generator.predict([img_input, zero_label])
        return result
    
    def save_model(self, dir_name='model'):
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            
        self.discriminator.save_weights(dir_name + '/rainclass_discriminator.hdf5')
        self.generator.save_weights(dir_name + '/rainclass_generator.hdf5')
        
    def load_model(self, dir_name='model'):
        discriminator_name = dir_name + '/rainclass_discriminator.hdf5'
        generator_name = dir_name + '/rainclass_generator.hdf5'
        
        if os.path.exists(discriminator_name) and os.path.isfile(discriminator_name):
            self.discriminator.load_model(discriminator_name)
        else:
            print('no discriminator model weights file in {0}'.format(discriminator_name))

        if os.path.exists(generator_name) and os.path.isfile(generator_name):
            self.generator.load_model(generator_name)
        else:
            print('no discriminator model weights file in {0}'.format(generator_name))
    