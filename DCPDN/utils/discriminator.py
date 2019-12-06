from .blocks import *
from compose import *
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Input, LeakyReLU, ZeroPadding2D
from tensorflow.keras.models import Model

__all__ = ['D']


def D(img_shape, num_filters=64):

    trans_input = Input((img_shape), name='trans_input')
    img_input = Input((img_shape), name='img_input')

    layers = compose(
        Concatenate(name='concat'),
        Conv2D(num_filters, kernel_size=4, strides=2,
               padding='same', use_bias=False, name='layer1'),
        UNetBlock(num_filters * 2, name='layer2', transposed=False,
                  bn=True, relu=False, dropout=False),
        UNetBlock(num_filters * 4, name='layer3', transposed=False,
                  bn=True, relu=False, dropout=False),

        LeakyReLU(alpha=0.2, name='layer4/leakyrelu'),
        Conv2D(num_filters * 8, kernel_size=4, strides=1,
               padding='valid', use_bias=False, name='layer4/conv'),
        ZeroPadding2D(name='layer4/zeropad'),
        BatchNormalization(name='layer4/bn'),

        LeakyReLU(alpha=0.2, name='layer5/leakyrelu'),
        Conv2D(1, kernel_size=4, strides=1, padding='valid',
               use_bias=False, name='layer5/conv'),
        ZeroPadding2D(name='layer5/zeropad'),
        Activation(activation='sigmoid', name='layer5/sigmoid'),
    )

    return Model(inputs=[trans_input, img_input], outputs=[layers([trans_input, img_input])])
