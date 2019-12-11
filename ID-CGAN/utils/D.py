from .blocks import *
from compose import *
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, LeakyReLU

__all__ = ['D']

multiplier_cal = lambda layer_idx: min(2 ** layer_idx, 8)

def D(input_num_channel=3, output_num_channel=3, num_filters=8, num_layers=3):
    
    layers = compose(
        Conv2D(num_filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')
    )
    
    for layer_idx in range(1, num_layers - 1):
        layers = compose(
            layers,
            UNetBlock_4_2(num_filters * multiplier_cal(layer_idx), 
                          name='layer%d' % (layer_idx + 1), 
                          transposed=False, bn=True, relu=False, dropout=False)
        )
    
    layers = compose(
        layers,
        UNetBlock_4_1(num_filters * multiplier_cal(num_layers - 1), 
                      name='layer%d' % num_layers, 
                      transposed=False, bn=True, relu=False, dropout=False)
    )
    
    layers = compose(
        layers,
        UNetBlock_4_1(1, name='layerfinal', transposed=False,
                      bn=False, relu=False, dropout=False),
        Activation('sigmoid', name='layerfinal/sigmoid')
    )
    
    return layers
