from base_blocks import *
from base_net_layers import *
from compose import *
from functools import partial
from math import log2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

__all__ = ['encoder_decoder', 'unet']

_base_unet_block = partial(unet_block, kernel_size=4, strides=2, bn=True)
EncoderBlock = partial(_base_unet_block, transposed=False, relu=False)
DecoderBlock = partial(_base_unet_block, transposed=True, relu=True)


def autoencoder(img_shape, ngf, include_top=True, name=None):

    layer_num = int(log2(min(img_shape[:2])))
    layers = autoencoder_layers(
        layer_num, img_shape[-1], ngf, include_top=include_top, name='autoencoder')
    img_input = Input(img_shape, name=name + 'img_input' if name else None)

    return Model(inputs=[img_input], outputs=[layers(img_input)])


def unet(img_shape, ngf, name=None):

    layer_num = int(log2(min(img_shape[:2])))
    layers = unet_layers(
        layer_num, img_shape[-1], ngf, include_top=True, name='unet')
    img_input = Input(img_shape, name=name + 'img_input' if name else None)

    return Model(inputs=[img_input], outputs=[layers(img_input)])
