from base_blocks import *
from tensorflow.keras.layers import AvgPool2D, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, ReLU, UpSampling2D
from functools import partial

__all__ = ['BottleneckBlock', 'TransitionBlock',
           'Sampling_Block', 'Concat_Samping_Block', 'UNetBlock']

DownSamplingLayer = partial(AvgPool2D)
UpSamplingLayer = partial(UpSampling2D, interpolation='nearest')

BottleneckBlock = partial(bottleneck_block, kernal_size_2=3)

TransitionBlock = partial(
    transition_block, transition_layer=UpSamplingLayer, transition_name='us')

Sampling_Block = partial(sampling_block, ds_layer=DownSamplingLayer,
                         us_layer=UpSamplingLayer, kernel_size=3)


def Concat_Samping_Block(pool_list, kernel_size=3, name=None):

    if name is not None:
        name = name + '/'

    return lambda x: Concatenate(name=name + 'concat' if name else None)(
        [Sampling_Block(pool, kernel_size=kernel_size, name=name + 'sample_{0}'.format(pool) if name else None)(x) for pool in pool_list] +
        [x]
    )


UNetBlock = partial(unet_block, kernel_size=4, strides=2)
