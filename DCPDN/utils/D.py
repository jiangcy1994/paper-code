from .blocks import *
from compose import *
from keras.layers import Activation, BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D

__all__ = ['D']

D = lambda num_filters=64: compose(
    Conv2D(num_filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1'),
    UNetBlock(num_filters * 2, name='layer2', transposed=False, bn=True, relu=False, dropout=False),
    UNetBlock(num_filters * 4, name='layer3', transposed=False, bn=True, relu=False, dropout=False),
    LeakyReLU(alpha=0.2, name='layer4.leakyrelu'),
    Conv2D(num_filters * 8, kernel_size=4, strides=1, padding='valid', use_bias=False, name='layer4.conv'),
    ZeroPadding2D(name='layer4.zeropad'),
    BatchNormalization(name='layer4.bn'),
    LeakyReLU(alpha=0.2, name='layer5.leakyrelu'),
    Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False, name='layer5.conv'),
    ZeroPadding2D(name='layer5.zeropad'),
    Activation(activation='sigmoid', name='layer5.sigmoid'),
)
