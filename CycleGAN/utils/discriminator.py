from base_blocks import *
from compose import *
from functools import partial
from tensorflow.keras.layers import Activation, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model

LayerBlock = partial(unet_block, kernel_size=4,
                     transposed=False, bn=True, relu=False)


def n_layers(img_shape, ndf, n_layers, use_sigmoid):

    img_input = Input(img_shape)
    x = Conv2D(ndf, kernel_size=4, strides=2,
               padding='same', name='layer0/conv')(img_input)

    for i in range(1, max_layers + 1):
        filters = ndf * min(2 ** layer, 8)
        if i == max_layers:
            x = LayerBlock(filters, strides=1,
                           name='layer{0}'.format(layer))(x)
        else:
            x = LayerBlock(filters, strides=2,
                           name='layer{0}'.format(layer))(x)

    x = LeakyReLU(alpha=0.2)(x)

    # final layer
    x = Conv2D(1, kernel_size=4, strides=1,
               name='layerfinal', padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)
    return Model(inputs=[img_input], outputs=[x])


def basic(img_shape, ndf, use_sigmoid):
    return _n_layers(img_shape, ndf, 3, use_sigmoid)


def imageGAN(img_shape, ndf, use_sigmoid):

    img_input = Input(img_shape)

    layers = compose(
        unet_block(ndf, 4, 2, 'layer1', transposed=False,
                   bn=False, relu=False),
        unet_block(ndf * 2, 4, 2, 'layer2',
                   transposed=False, bn=True, relu=False),
        unet_block(ndf * 4, 4, 2, 'layer3',
                   transposed=False, bn=True, relu=False),
        unet_block(ndf * 8, 4, 2, 'layer4',
                   transposed=False, bn=True, relu=False),
        unet_block(ndf * 8, 4, 2, 'layer5',
                   transposed=False, bn=True, relu=False),
        unet_block(ndf * 8, 4, 2, 'layer6',
                   transposed=False, bn=True, relu=False),
        Conv2D(1, kernel_size=4, strides=2, padding='same', name='layer7')
    )

    if use_sigmoid:
        layers = compose(
            layers,
            Activation('sigmoid')
        )

    output = layer(img_input)
    return Model(inputs=[img_input], outputs=[output])


def pixelGAN(img_shape, ndf, use_sigmoid):

    img_input = Input(img_shape)

    layers = compose(
        unet_block(ndf, 1, 1, 'layer1', transposed=False,
                   bn=False, relu=False),
        unet_block(ndf * 2, 1, 1, 'layer2',
                   transposed=False, bn=True, relu=False),
        Conv2D(1, kernel_size=4, strides=2, padding='same', name='layer7')
    )

    if use_sigmoid:
        layers = compose(
            layers,
            Activation('sigmoid')
        )

    output = layer(img_input)
    return Model(inputs=[img_input], outputs=[output])
