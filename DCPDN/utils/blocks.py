from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, ReLU, UpSampling2D
from keras.models import Model
from functools import partial

from .compose import compose

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

def UNetBlock(x, out_channel, name, transposed=False, bn=False, relu=True, dropout=False):
    if relu:
        out = ReLU(name='%s.relu' % name)(x)
    else:
        out = LeakyReLU(alpha=0.2, name='%s.leakyrelu' % name)(x)
    if not transposed:
        out = Conv2D(out_channel, kernel_size=4, strides=2, padding='same', use_bias=False, name='%s.conv' % name)(out)
    else:
        out = Conv2DTranspose(out_channel, kernel_size=4, strides=2, padding='same', use_bias=False, name='%s.tconv' % name)(out)
    if bn:
        out = BatchNormalization(name='%s.bn' % name)(out)
    if dropout:
        out = Dropout(0.5, name='%s.dropout' % name)(out)
    return out

def Sampling_Block(pool, kernel_size=3):
    return compose(
        AveragePooling2D(pool),
        Conv2D(1, kernel_size=kernel_size, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(pool)
    )
