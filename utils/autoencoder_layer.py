from base_blocks import *
from compose import *
from functools import partial
from tensorflow.keras.layers import Activation, Conv2D, Conv2DTranspose, ReLU

__all__ = ['autoencoder_layers']

_base_unet_block = partial(unet_block, kernel_size=4, strides=2, bn=True)
EncoderBlock = partial(_base_unet_block, transposed=False, relu=False)
DecoderBlock = partial(_base_unet_block, transposed=True, relu=True)


def autoencoder_layers(layer_num, output_channel, nf, include_top=True, name=None):

    assert layer_num >= 2
    def multiplier(x): return min(x ** 2, 8)

    if name is not None:
        name = name + '/'

    # First Encoder Layers
    encoder_layers = Conv2D(nf, kernel_size=4, strides=2, padding='same',
                            name=name + 'encoder_block1/conv' if name else None)

    # Last Decoder Layers
    decoder_layers = compose(
        ReLU(name=name + 'decoder_block1/relu' if name else None),
        Conv2DTranspose(output_channel, kernel_size=4, strides=2, padding='same',
                        name=name + 'decoder_block1/tconv' if name else None)
    )

    # Inter Layers
    for i in range(1, layer_num):

        encoder_layers = compose(
            encoder_layers,
            EncoderBlock(nf * multiplier(i), name=name +
                         'encoder_block{0}'.format(i + 1) if name else None),
        )

        decoder_layers = compose(
            DecoderBlock(nf * multiplier(i - 1), name=name +
                         'decoder_block{0}'.format(i + 1) if name else None),
            decoder_layers
        )

    layers = compose(
        encoder_layers,
        decoder_layers
    )

    if include_top:
        layers = compose(
            layers,
            Activation('tanh', name=name +
                       'output_block/tanh' if name else None)
        )

    return layers
