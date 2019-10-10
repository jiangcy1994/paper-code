from keras.layers import AvgPool2D, BatchNormalization, Conv2D, LeakyReLU, ReLU, UpSampling2D
from keras.models import Model
from functools import partial

from .compose import compose

def conv_block(in_dim,out_dim):
    return compose(
        Conv2D(in_dim, kernel_size=3, strides=1, padding='same'),
        ELU(),
        Conv2D(in_dim, kernel_size=3, strides=1, padding='same'),
        ELU(),
        Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
        AvgPool2D()
    )

def deconv_block(in_dim,out_dim):
    return compose(
        Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
        ELU(),
        Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
        UpSampling2D()
    )

def BottleneckBlock(x, in_planes, out_planes, dropRate=0.0):
    inter_planes = out_planes * 4
    out = compose(
        BatchNormalization(),
        ReLU(),
        Conv2D(inter_planes, kernel_size=1, strides=1, padding='same', use_bias=False)
    )(x)
    if dropRate:
        out = Dropout(dropRate)(out)
    out = compose(
        BatchNormalization(),
        ReLU(),
        Conv2D(out_planes, kernel_size=3, strides=1, padding='same', use_bias=False)
    )(out)
    if dropRate:
        out = Dropout(dropRate)(out)
    return out

def TransitionBlock(x, in_planes, out_planes, dropRate=0.0):
    out = compose(
        BatchNormalization(),
        ReLU(),
        Conv2D(out_planes, kernel_size=1, strides=1, padding='same', use_bias=False)
    )(x)
    if dropRate:
        out = Dropout(dropRate)(out)
    out = UpSampling2D()(out)
    return out

def Dense_Block():
    return compose(
        Conv2D(1, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(1)
    )
