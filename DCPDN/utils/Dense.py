from .blocks import *
from keras.applications import densenet
from keras.layers import Activation, Concatenate, Conv2D, Input
from keras.models import Model

__all__ = ['Dense']

bottleneck_transition_block = lambda bottleneck_filter, transition_filter: compose(
    BottleneckBlock(bottleneck_filter),
    TransitionBlock(transition_filter)
)

def Dense(img_shape=(256,256,3)):
    
    dense121_model = densenet.DenseNet121(include_top=False, weights='imagenet', img_shape)
    dense121_model.layers.remove(dense121_model.layers[1])

    x0 = dense121_model.input
    x1 = dense121_model.layers[52 - 1].output
    x2 = dense121_model.layers[140 - 1].output
    x3 = dense121_model.layers[312 - 1].output
    
    x4 = Concatenate()([
        bottleneck_transition_block(256, 128)(x3),
        x2
    ])
    
    x5 = Concatenate()([
        bottleneck_transition_block(256, 128)(x4), 
        x1
    ])
    
    x8 = Concatenate()([
        compose(
            bottleneck_transition_block(128, 64),
            bottleneck_transition_block(64, 32),
            bottleneck_transition_block(32, 16)
        )(x5)
        x0
    ])
    
    x9 = compose(
        Conv2D(20, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )(x8)
    
    sampling_block = lambda pool: Sampling_Block(pool, kernel_size=1)
    
    dehaze = compose(
        Concatenate(),
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )([
        sampling_block(32)(x9),
        sampling_block(16)(x9),
        sampling_block(8)(x9),
        sampling_block(4)(x9),
        x9]
    )
    
    return Model(inputs=[x0], outputs=[dehaze])
