from base_blocks import *
from compose import *
from functools import partial
from tensorflow.keras.layers import Activation, Conv2D, Conv2DTranspose, ReLU

__all__ = ['unet_layers']

_base_unet_block = partial(unet_block, kernel_size=4, strides=2, bn=True)
EncoderBlock = partial(_base_unet_block, transposed=False, relu=False)
DecoderBlock = partial(_base_unet_block, transposed=True, relu=True)


def unet_layers(layer_num, output_channel, nf, include_top=True, name=None):

    assert layer_num >= 2
    def multiplier(x): return min(x ** 2, 8)

    if name is not None:
        name = name + '/'

    # Innermost Layers
    layers = compose(
        EncoderBlock(nf * multiplier(layer_num),
                     name=name + 'encoder_block{0}'.format(layer_num) if name else None),
        DecoderBlock(nf * multiplier(layer_num), dropout=True,
                     name=name + 'decoder_block{0}'.format(layer_num) if name else None)
    )

    # Inter Layers
    for i in range(1, layer_num - 1):

        layer_idx = layer_num - i
        layers = compose(
            EncoderBlock(nf * multiplier(layer_idx),
                         name=name + 'encoder_block{0}'.format(layer_idx) if name else None),
            skip_concat(layers, name +
                        'concat{0}'.format(layer_idx) if name else None),
            DecoderBlock(nf * multiplier(layer_idx), dropout=True if i < 3 else False,
                         name=name + 'decoder_block{0}'.format(layer_idx) if name else None)
        )

    # Outermost Layers
    layers = compose(
        Conv2D(nf, kernel_size=4, strides=2, padding='same', use_bias=False,
               name=name + 'encoder_block1/conv' if name else None),
        skip_concat(layers, name + 'concat{0}'.format(1) if name else None),
        ReLU(name=name + 'decoder_block1/relu' if name else None),
        Conv2DTranspose(output_channel, kernel_size=4, strides=2, padding='same',
                        name=name + 'decoder_block1/tconv' if name else None)
    )

    if include_top:
        layers = compose(
            layers,
            Activation('tanh', name=name +
                       'output_block/tanh' if name else None)
        )

    return layers
