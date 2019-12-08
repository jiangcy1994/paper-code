from base_blocks import *
from tensorflow.keras.layers import AvgPool2D, UpSampling2D
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

Concat_Samping_Block = partial(concat_samping_block, ds_layer=DownSamplingLayer,
                         us_layer=UpSamplingLayer, kernel_size=3)

UNetBlock = partial(unet_block, kernel_size=4, strides=2)
