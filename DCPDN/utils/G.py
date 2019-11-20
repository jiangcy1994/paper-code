from .blocks import *
from compose import *
from keras.layers import Activation, Concatenate, Conv2D, Input
from keras.models import Model

__all__ = ['G', 'G2']

conv_filters_multipliers = [1, 2, 4, 8, 8, 8, 8, 8]
conv_transpose_filters_multipliers = [8, 8, 8, 8, 4, 2, 1]

def G_base(inputs, output_num_channel=3, num_filters=64):
    outs = []
    x = Conv2D(num_filters * conv_filters_multipliers[0], kernel_size=4, strides=2, 
               padding='same', use_bias=False, name='layer1')(inputs)
    outs.append(x)
    
    for layer_idx, multiplier in enumerate(conv_filters_multipliers[1:]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock(num_filters * multiplier, name=name, transposed=False, 
                      bn=True, relu=False, dropout=False)(x)
        outs.append(x)

    x = UNetBlock(num_filters * conv_transpose_filters_multipliers[0], name='dlayer8', transposed=True,
                  bn=False, relu=True, dropout=True)(x)

    for layer_idx, multiplier in enumerate(conv_transpose_filters_multipliers[1:]):
        layer_idx = 7 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = Concatenate()([x, outs[layer_idx - 1]])
        x = UNetBlock(num_filters * multiplier, name=name, transposed=True, 
                      bn=True, relu=True, dropout=False)(x)

    x = compose(
        Concatenate(),
        UNetBlock(output_num_channel, name='dlayer1', transposed=True, bn=False, relu=True, dropout=False)
    )([x, outs[0]])
    return x
    

def G(img_shape=(256, 256), input_num_channel=3, output_num_channel=3, num_filters=64):
    img_input = Input(img_shape + (input_num_channel,))
    x = G_base(img_input, 20, num_filters)
    
    dehaze = compose(
        Concatenate(),
        Conv2D(output_num_channel, kernel_size=3, strides=1, 
               padding='same', use_bias=False, name='layerfinal.conv'),
        Activation('tanh', name='layerfinal.tanh')
    )([
        Sampling_Block(16)(x),
        Sampling_Block(8)(x),
        Sampling_Block(4)(x),
        Sampling_Block(2)(x),
        x
    ])

    return Model(inputs=[img_input], outputs=[dehaze])

def G2(img_shape=(256, 256), input_num_channel=3, output_num_channel=3, num_filters=8):
    img_input = Input(img_shape + (input_num_channel,))

    output = compose(
        Activation('tanh', name='dlayer1.tanh')
    )(G_base(img_input, output_num_channel, num_filters))

    return Model(inputs=[img_input], outputs=[output])
