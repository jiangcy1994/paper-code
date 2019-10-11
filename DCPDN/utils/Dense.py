from keras.applications import densenet
from keras.layers import Activation, Concatenate, Conv2D, Input
from keras.models import Model

from .blocks import BottleneckBlock, TransitionBlock, Sampling_Block

__all__ = ['Dense']

def Dense():
    dense121_model = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(256,256,3))
    dense121_model.layers.remove(dense121_model.layers[1])

    x0 = dense121_model.input
    
    x1 = dense121_model.layers[52 - 1].output
    x2 = dense121_model.layers[140 - 1].output
    x3 = dense121_model.layers[312 - 1].output
    
    x4 = TransitionBlock(BottleneckBlock(x3, 512, 256), 768, 128)
    x4 = Concatenate()([x4, x2])
    
    x5 = TransitionBlock(BottleneckBlock(x4, 384, 256), 640, 128)
    x5 = Concatenate()([x5, x1])
    
    x6 = TransitionBlock(BottleneckBlock(x5, 256, 128), 384, 64)
    x7 = TransitionBlock(BottleneckBlock(x6, 64, 64), 128, 32)
    x8 = TransitionBlock(BottleneckBlock(x7, 32, 32), 64, 16)
    x8 = Concatenate()([x8, x0])
    
    x9 = Activation('tanh')(Conv2D(20, kernel_size=3, strides=1, padding='same')(x8))
    
    x1010 = Sampling_Block(32, kernel_size=1)(x9)
    x1020 = Sampling_Block(16, kernel_size=1)(x9)
    x1030 = Sampling_Block(8, kernel_size=1)(x9)
    x1040 = Sampling_Block(4, kernel_size=1)(x9)
    
    dehaze = Concatenate()([x1010, x1020, x1030, x1040, x9])
    dehaze = Activation('tanh')(Conv2D(3, kernel_size=3, strides=1, padding='same')(dehaze))
    
    return Model(inputs=[x0], outputs=[dehaze])
