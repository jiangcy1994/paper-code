from keras.layers import AvgPool2D, BatchNormalization, Conv2D, LeakyReLU, ReLU, UpSampling2D
from keras.models import Model
from functools import partial

from compose import compose

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
        
def _bottleneck_block(x, in_planes, out_planes, _kernal_size_2, dropRate=0.0):
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
        Conv2D(out_planes, kernel_size=_kernal_size_2, strides=1, padding='same', use_bias=False)
    )(out)
    if dropRate:
        out = Dropout(dropRate)(out)
    return out

BottleneckBlock = partial(_bottleneck_block, _kernal_size_2=3)
BottleneckBlock1 = partial(_bottleneck_block, _kernal_size_2=5)
BottleneckBlock2 = partial(_bottleneck_block, _kernal_size_2=7)

def _transition_block(x, in_planes, out_planes, _transition_func=None, dropRate=0.0):
    out = compose(
        BatchNormalization(),
        ReLU(),
        Conv2D(out_planes, kernel_size=1, strides=1, padding='same', use_bias=False)
    )(x)
    if dropRate:
        out = Dropout(dropRate)(out)
    if _transition_func:
        out = _transition_func(out)
    return out

TransitionBlock = partial(_transition_block, _transition_func=UpSampling2D())
TransitionBlock1 = partial(_transition_block, _transition_func=AvgPool2D())
TransitionBlock3 = partial(_transition_block)

def Dense_Block():
    return compose(
        Conv2D(1, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(1)
    )
