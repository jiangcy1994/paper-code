from base_blocks import *
from compose import *
from functools import partial
from tensorflow.keras.layers import Activation, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model

__all__ = ['basic', 'n_layers']

LayerBlock = partial(unet_block, kernel_size=4,
                     transposed=False, bn=True, relu=False)


def n_layers(img_shape, ndf, n_layers, use_sigmoid):

    img_input = Input(img_shape)
    x = Conv2D(ndf, kernel_size=4, strides=2,
               padding='same', name='layer0/conv')(img_input)

    for i in range(1, n_layers + 1):
        filters = ndf * min(2 ** i, 8)
        if i == n_layers:
            x = LayerBlock(filters, strides=1,
                           name='layer{0}'.format(i))(x)
        else:
            x = LayerBlock(filters, strides=2,
                           name='layer{0}'.format(i))(x)

    # final layer
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=4, strides=1,
               name='layerfinal', padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)
    return Model(inputs=[img_input], outputs=[x])


def basic(img_shape, ndf, use_sigmoid):
    return n_layers(img_shape, ndf, 3, use_sigmoid)
