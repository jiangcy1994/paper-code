from compose import *
from keras.applications import vgg19
from keras.layers import AvgPool2D, BatchNormalization, Conv2D, Dense, Flatten, Input, ReLU, UpSampling2D
from keras.models import Model

__all__ = ['VGG19ca']

def VGG19ca(img_shape=(128,128,3)):
    vgg19_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=img_shape)

    # Build Model
    inputs = vgg19_model.input
    
    # Blocks in VGG19
    blocks_vgg19 = compose(
        Conv2D(64, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(epsilon=1e-05, momentum=0.1),
        ReLU(),
        Conv2D(24, kernel_size=3, strides=1, padding='same'),
        ReLU(),
        AvgPool2D(7)
    )
    
    #Dense
    blocks_dense = compose(
        Flatten(),
        Dense(512),
        ReLU(),
        Dense(4)
    )
    
    model = Model(
        inputs=[inputs], 
        outputs=[compose(blocks_vgg19, blocks_dense)(inputs)]
    )
    
    # Transfer Weight
    model.layers[1].set_weights(vgg19_model.layers[1].get_weights())
    return model
