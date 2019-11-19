from base_blocks import *
from compose import *
from keras.layers import AvgPool2D, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, ReLU, UpSampling2D
from functools import partial

__all__ = ['BottleneckBlock', 'TransitionBlock', 'Sampling_Block', 'UNetBlock']

DownSamplingLayer = partial(AvgPool2D)
UpSamplingLayer = partial(UpSampling2D, interpolation='nearest')

BottleneckBlock = partial(bottleneck_block, kernal_size_2=3)

TransitionBlock = partial(transition_block, transition_layer=UpSamplingLayer, transition_name='us')

Sampling_Block = partial(sampling_block, ds_layer=DownSamplingLayer, us_layer=UpSamplingLayer, kernel_size=3)


def UNetBlock(filters, name, transposed=False, bn=False, relu=True, dropout=False):
    
    if relu:
        block_1 = compose(
            ReLU(name='%s.relu' % name)
        )
    else:
        block_1 = compose(
            LeakyReLU(alpha=0.2, name='%s.leakyrelu' % name)
        )
        
    if not transposed:
        block_2 = compose(
            Conv2D(filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='%s.conv' % name)
        )
    else:
        block_2 = compose(
            Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', use_bias=False, name='%s.tconv' % name)
        )
    
    layers = compose(block_1, block_2)
    
    if bn:
        layers = compose(
            layers,
            BatchNormalization(name='%s.bn' % name)
        )
    
    if dropout:
        layers = compose(
            layers,
            Dropout(0.5, name='%s.dropout' % name)
        )
        
    return layers
