from .blocks import *
from base_net_layers import *
from compose import *
from math import log2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Add, AvgPool2D, Concatenate, Conv2D, Input, Lambda, LeakyReLU, Multiply, Subtract, UpSampling2D
from tensorflow.keras.models import Model

__all__ = ['Dehaze']


def transmission_map_generator(img_shape, output_channel=3, ngf=64):

    layer_num = int(log2(min(img_shape[:2])))
    layers = compose(
        unet_layers(layer_num, 20, ngf,
                    include_top=False, name='trans'),
        Concat_Samping_Block([16, 8, 4, 2], name='trans'),
        Conv2D(output_channel, kernel_size=3, strides=1,
               padding='same', use_bias=False, name='trans/output_block/conv'),
        Activation('tanh', name='trans/output_block/tanh')
    )

    return layers


def atmospheric_light_generator(img_shape, output_channel=3, ngf=8):

    layer_num = int(log2(min(img_shape[:2])))
    layers = unet_layers(
        layer_num, output_channel, ngf, include_top=True, name='atmos')

    return layers


def Dehaze(img_shape=(256, 256, 3)):

    img_input = Input(img_shape, name='img_input')
    trans = transmission_map_generator(img_shape)(img_input)
    atmos = atmospheric_light_generator(img_shape)(img_input)

    # $trans_{reciprocal} = \frac{1}{trans + 10^{-10}}$
    trans_reciprocal = Lambda(
        function=lambda x: 1 / (K.abs(x) + 10**-10))(trans)

    atmos = compose(
        AvgPool2D(),
        LeakyReLU(0.2),
        UpSampling2D()
    )(atmos)

    # $dehaze = (input - atmos) \times trans^{-1} + atmos$
    dehaze = Subtract()([img_input, atmos])
    dehaze = Multiply()([dehaze, trans_reciprocal])
    dehaze = Add()([dehaze, atmos])

    dehaze = compose(
        Concatenate(),
        Conv2D(6, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(20, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Concat_Samping_Block([32, 16, 8, 4], kernel_size=1),
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )([dehaze, img_input])

    return Model(inputs=[img_input], outputs=[dehaze, trans, atmos])
