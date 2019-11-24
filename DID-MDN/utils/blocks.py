from base_blocks import *
from compose import *
from functools import partial
from keras.layers import AvgPool2D, Conv2D, LeakyReLU, UpSampling2D

__all__ = ['BottleneckBlock_3', 'BottleneckBlock_5', 'BottleneckBlock_7', 
           'TransitionBlock_Up', 'TransitionBlock_Down', 'TransitionBlock_Plain', 
           'UpSampling_Block', 'Sampling_Block']

DownSamplingLayer = partial(AvgPool2D)
UpSamplingLayer = partial(UpSampling2D, interpolation='nearest')

BottleneckBlock_3 = partial(bottleneck_block, kernal_size_2=3)
BottleneckBlock_5 = partial(bottleneck_block, kernal_size_2=5)
BottleneckBlock_7 = partial(bottleneck_block, kernal_size_2=7)

TransitionBlock_Down = partial(transition_block, transition_layer=DownSamplingLayer, transition_name='down')
TransitionBlock_Plain = partial(transition_block)
TransitionBlock_Up = partial(transition_block, transition_layer=UpSamplingLayer, transition_name='up')

def UpSampling_Block(up_sample_size=1, name=None):
    return compose(
        Conv2D(1, kernel_size=3, strides=1, padding='same', 
               name=name + '/conv2d' if name else None),
        LeakyReLU(alpha=0.2, name=name + '/lrelu' if name else None),
        UpSampling2D(up_sample_size, name=name + '/us' if name else None)
    )

Sampling_Block = partial(sampling_block, ds_layer=DownSamplingLayer, us_layer=UpSamplingLayer, kernel_size=1)
