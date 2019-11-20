from base_blocks import *
from keras.layers import AvgPool2D, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, ReLU, UpSampling2D
from functools import partial

__all__ = ['BottleneckBlock', 'TransitionBlock', 'Sampling_Block', 
           'UNetBlock_3_1', 'UNetBlock_4_1', 'UNetBlock_4_2']

DownSamplingLayer = partial(AvgPool2D)
UpSamplingLayer = partial(UpSampling2D, interpolation='nearest')

BottleneckBlock = partial(bottleneck_block, kernal_size_2=3)

TransitionBlock = partial(transition_block, transition_layer=UpSamplingLayer, transition_name='us')

Sampling_Block = partial(sampling_block, ds_layer=DownSamplingLayer, us_layer=UpSamplingLayer, kernel_size=3)

UNetBlock_3_1 = partial(unet_block, kernel_size=3, strides=1)
UNetBlock_4_1 = partial(unet_block, kernel_size=4, strides=1)
UNetBlock_4_2 = partial(unet_block, kernel_size=4, strides=2)
