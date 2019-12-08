from base_blocks import *
from compose import *
from functools import partial
from tensorflow.keras.layers import AvgPool2D, Conv2D, LeakyReLU, UpSampling2D

__all__ = ['BottleneckBlock_3', 'BottleneckBlock_5', 'BottleneckBlock_7',
           'TransitionBlock_Up', 'TransitionBlock_Down', 'TransitionBlock_Plain',
           'UpSampling_Block', 'Sampling_Block', 'Concat_Samping_Block', 'DB_Blocks_Gen', 'US_Blocks_Gen']

DownSamplingLayer = partial(AvgPool2D)
UpSamplingLayer = partial(UpSampling2D, interpolation='nearest')

BottleneckBlock_3 = partial(bottleneck_block, kernal_size_2=3)
BottleneckBlock_5 = partial(bottleneck_block, kernal_size_2=5)
BottleneckBlock_7 = partial(bottleneck_block, kernal_size_2=7)

TransitionBlock_Down = partial(
    transition_block, transition_layer=DownSamplingLayer, transition_name='down')
TransitionBlock_Plain = partial(transition_block)
TransitionBlock_Up = partial(
    transition_block, transition_layer=UpSamplingLayer, transition_name='up')


def UpSampling_Block(up_sample_size=1, name=None):
    return compose(
        Conv2D(1, kernel_size=3, strides=1, padding='same',
               name=name + '/conv2d' if name else None),
        LeakyReLU(alpha=0.2, name=name + '/lrelu' if name else None),
        UpSampling2D(up_sample_size, name=name + '/us' if name else None)
    )


Sampling_Block = partial(sampling_block, ds_layer=DownSamplingLayer,
                         us_layer=UpSamplingLayer, kernel_size=1)

Concat_Samping_Block = partial(concat_samping_block, ds_layer=DownSamplingLayer,
                               us_layer=UpSamplingLayer, kernel_size=1)


def DB_Blocks_Gen(_bn_block, _bn_filters, _tr_blocks, _tr_filters, name=None):

    if name is not None:
        name = name + '/'

    blocks = []
    for i, (_bn_filter, _tr_block, _tr_filter) in enumerate(zip(_bn_filters, _tr_blocks, _tr_filters)):

        block_id_str = 'db_{0}'.format(i + 1)
        if _tr_block is TransitionBlock_Down:
            tr_str = '/td'
        if _tr_block is TransitionBlock_Plain:
            tr_str = '/tp'
        if _tr_block is TransitionBlock_Up:
            tr_str = '/tu'

        blocks.append(
            compose(
                _bn_block(_bn_filter, name=name +
                          block_id_str + '/bn' if name else None),
                _tr_block(_tr_filter, name=name +
                          block_id_str + tr_str if name else None)
            )
        )
    return blocks


def US_Blocks_Gen(_us_pools, name=None):

    if name is not None:
        name = name + '/'

    return [UpSampling_Block(pool,
                             name=name + 'us_{0}'.format(i + 1) if name else None)
            for i, pool in enumerate(_us_pools)]
