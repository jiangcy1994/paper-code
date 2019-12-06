from compose import *
from functools import partial
from tensorflow.keras.layers import AvgPool2D, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, ReLU, UpSampling2D

__all__ = ['bottleneck_block', 'transition_block',
           'sampling_block', 'unet_block', 'skip_concat']


def bottleneck_block(out_filters, kernal_size_2, dropRate=0.0, name=None):

    inter_filters = out_filters * 4
    if name is not None:
        name = name + '/'

    layer_1 = compose(
        BatchNormalization(name=name + 'bn1' if name else None),
        ReLU(name=name + 'relu1' if name else None),
        Conv2D(inter_filters, kernel_size=1, strides=1, padding='same', use_bias=False,
               name=name + 'conv2d1' if name else None)
    )

    if dropRate:
        layer_1 = compose(
            layer_1,
            Dropout(dropRate, name=name + 'dropout1' if name else None)
        )

    layer_2 = compose(
        BatchNormalization(name=name + 'bn2' if name else None),
        ReLU(name=name + 'relu2' if name else None),
        Conv2D(out_filters, kernel_size=kernal_size_2, strides=1, padding='same', use_bias=False,
               name=name + 'conv2d2' if name else None)
    )

    if dropRate:
        layer_2 = compose(
            layer_2,
            Dropout(dropRate, name=name + 'dropout2' if name else None)
        )

    return compose(layer_1, layer_2)


def transition_block(out_filters, transition_layer=None, transition_name='trans', dropRate=0.0, name=None):

    if name is not None:
        name = name + '/'

    layers = compose(
        BatchNormalization(name=name + 'bn' if name else None),
        ReLU(name=name + 'relu' if name else None),
        Conv2D(out_filters, kernel_size=1, strides=1, padding='same', use_bias=False,
               name=name + 'conv' if name else None)
    )

    if dropRate:
        layers = compose(
            layers,
            Dropout(dropRate, name=name + 'dropout' if name else None)
        )

    if transition_layer:
        layers = compose(
            layers,
            transition_layer(name=name + transition_name if name else None)
        )

    return layers


def sampling_block(pool, ds_layer, us_layer, kernel_size=1, name=None):

    if name is not None:
        name = name + '/'

    layers = compose(
        ds_layer(pool, name=name + '/ds' if name else None),
        Conv2D(1, kernel_size=kernel_size, strides=1, padding='same',
               name=name + '/conv2d' if name else None),
        LeakyReLU(alpha=0.2, name=name + '/lrelu' if name else None),
        us_layer(pool, name=name + '/us' if name else None)
    )

    return layers


def unet_block(filters, kernel_size, strides, name, transposed=False, bn=False, relu=True, dropout=False):

    if name is not None:
        name = name + '/'

    if relu:
        block_1 = compose(
            ReLU(name=name + 'relu' if name else None)
        )
    else:
        block_1 = compose(
            LeakyReLU(alpha=0.2, name=name + 'lrelu' if name else None)
        )

    if not transposed:
        block_2 = compose(
            Conv2D(filters, kernel_size=kernel_size, strides=strides,
                   padding='same', use_bias=False, name=name + 'conv' if name else None)
        )
    else:
        block_2 = compose(
            Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                            padding='same', use_bias=False, name=name + 'tconv' if name else None)
        )

    layers = compose(block_1, block_2)

    if bn:
        layers = compose(
            layers,
            BatchNormalization(name=name + 'bn' if name else None)
        )

    if dropout:
        layers = compose(
            layers,
            Dropout(0.5, name=name + 'dropout' if name else None)
        )

    return layers


def skip_concat(f, name=None):
    return lambda x: Concatenate(name=name)([x, f(x)])
