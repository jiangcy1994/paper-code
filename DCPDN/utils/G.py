from .blocks import *
from compose import *
from keras.layers import Activation, Concatenate, Conv2D, Input
from keras.models import Model

__all__ = ['G', 'G2']

def G(img_shape=(256, 256), input_num_channel=3, output_num_channel=3, num_filters=64):
    
    img_input = Input(img_shape + (input_num_channel,))
    x = img_input
    outs = []
    
    x = Conv2D(num_filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')(x)
    outs.append(x)
    
    for layer_idx, multiplier in enumerate([2, 4, 8, 8, 8, 8, 8]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock(num_filters * multiplier, name, transposed=False, bn=True, relu=False, dropout=False)(x)
        outs.append(x)

    x = UNetBlock(num_filters * 8, 'dlayer8', transposed=True, bn=False, relu=True, dropout=True)(x)
multiplier
    for layer_idx, multiplier in enumerate([8, 8, 8, 4, 2, 1]):
        layer_idx = 7 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = Concatenate()([x, outs[layer_idx - 1]])
        x = UNetBlock(num_filters * multiplier, name, transposed=True, bn=True, relu=True, dropout=False)(x)

    x = Concatenate()([x, outs[0]])
    x = UNetBlock(20, 'dlayer%d' % 1, transposed=True, bn=False, relu=True, dropout=False)(x)
    
    dehaze = compose(
        Concatenate(),
        Conv2D(output_num_channel, kernel_size=3, strides=1, padding='same', use_bias=False, name='layerfinal.conv'),
        Activation('tanh', name='layerfinal.tanh')
    )([
        Sampling_Block(16)(x),
        Sampling_Block(8)(x),
        Sampling_Block(4)(x),
        Sampling_Block(2)(x),
        x
    ])

    model = Model(inputs=[img_input], outputs=[dehaze])
    return model

def G2(img_shape=(256, 256), input_num_channel=3, output_num_channel=3, num_filters=8):
    
    img_input = Input(img_shape + (input_num_channel,))
    x = img_input
    outs = []
    
    x = Conv2D(num_filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')(x)
    outs.append(x)
    
    for layer_idx, multiplier in enumerate([2, 4, 8, 8, 8, 8, 8]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock(num_filters * multiplier, name, transposed=False, bn=True, relu=False, dropout=False)(x)
        outs.append(x)

    x = UNetBlock(num_filters * 8, 'dlayer8', transposed=True, bn=False, relu=True, dropout=True)(x)

    for layer_idx, multiplier in enumerate([8, 8, 8, 4, 2, 1]):
        layer_idx = 7 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = Concatenate()([x, outs[layer_idx - 1]])
        x = UNetBlock(num_filters * multiplier, name, transposed=True, bn=True, relu=True, dropout=False)(x)

    output = compose(
        Concatenate(),
        UNetBlock(output_num_channel, 'dlayer1', transposed=True, bn=False, relu=True, dropout=False),
        Activation('tanh', name='dlayer1.tanh')
    )([x, outs[0]])

    model = Model(inputs=[img_input], outputs=[output])
    return model
