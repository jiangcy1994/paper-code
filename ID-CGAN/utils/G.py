from .blocks import *
from compose import *
from tensorflow.keras.layers import Activation, Add, Conv2D, Input
from tensorflow.keras.models import Model

__all__ = ['G']

conv_filters_multipliers = [1, 1, 1, 1, 0.5, 0]
conv_transpose_filters_multipliers = [0.5, 1, 1, 1, 1]

def G_base(inputs, output_num_channel=3, num_filters=64):
    outs = []
    x = Conv2D(num_filters * conv_filters_multipliers[0], kernel_size=3, strides=1, 
               padding='same', use_bias=False, name='layer1')(inputs)
    outs.append(x)
    
    for layer_idx, multiplier in enumerate(conv_filters_multipliers[1:]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock_3_1(int(num_filters * multiplier) if multiplier != 0 else 1, name=name, transposed=False, 
                          bn=True, relu=False, dropout=False)(x)
        outs.append(x)

    x = UNetBlock_3_1(int(num_filters * conv_transpose_filters_multipliers[0]), name='dlayer6', transposed=True,
                      bn=False, relu=True, dropout=True)(x)

    for layer_idx, multiplier in enumerate(conv_transpose_filters_multipliers[1:]):
        layer_idx = 5 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = UNetBlock_3_1(num_filters * multiplier, name=name, transposed=True, 
                          bn=True, relu=True, dropout=False)(x)
        if layer_idx in [3, 5]:
            x = Add()([x, outs[layer_idx - 2]])

    x = compose(
        UNetBlock_3_1(output_num_channel, name='dlayer1', transposed=True, bn=False, relu=True, dropout=False)
    )(x)
    return x

def G(img_shape=(256, 256), input_num_channel=3, output_num_channel=3, num_filters=8):
    img_input = Input(img_shape + (input_num_channel,))

    output = compose(
        Activation('tanh', name='dlayer1/tanh')
    )(G_base(img_input, output_num_channel, num_filters))

    return Model(inputs=[img_input], outputs=[output])
