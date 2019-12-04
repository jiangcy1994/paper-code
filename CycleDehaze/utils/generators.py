from base_blocks import *
from compose import *
from functools import partial
from layers import *
from math import log2
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Input, ReLU
from tensorflow.keras.models import Model

__all__ = ['encoder_decoder', 'unet', 'resnet_6blocks', 'resnet_9blocks']

_base_unet_block = partial(unet_block, kernel_size=4, strides=2, bn=True)
EncoderBlock = partial(_base_unet_block, transposed=False, relu=False)
DecoderBlock = partial(_base_unet_block, transposed=True, relu=True)


def encoder_decoder(img_shape, ngf, name=None):

    if name is not None:
        name = name + '/'

    img_input = Input(img_shape, name=name + 'input' if name else None)

    output = compose(
        Conv2D(ngf, kernel_size=4, strides=2, padding='same',
               name=name + 'encoder_block1/conv' if name else None),
        EncoderBlock(ngf * 2, name=name + 'encoder_block2' if name else None),
        EncoderBlock(ngf * 4, name=name + 'encoder_block3' if name else None),
        EncoderBlock(ngf * 8, name=name + 'encoder_block4' if name else None),
        EncoderBlock(ngf * 8, name=name + 'encoder_block5' if name else None),
        EncoderBlock(ngf * 8, name=name + 'encoder_block6' if name else None),
        EncoderBlock(ngf * 8, name=name + 'encoder_block7' if name else None),
        EncoderBlock(ngf * 8, name=name + 'encoder_block8' if name else None),
        DecoderBlock(ngf * 8, name=name +
                     'decoder_block1' if name else None, dropout=True),
        DecoderBlock(ngf * 8, name=name +
                     'decoder_block2' if name else None, dropout=True),
        DecoderBlock(ngf * 8, name=name +
                     'decoder_block3' if name else None, dropout=True),
        DecoderBlock(ngf * 8, name=name + 'decoder_block4' if name else None),
        DecoderBlock(ngf * 4, name=name + 'decoder_block5' if name else None),
        DecoderBlock(ngf * 2, name=name + 'decoder_block6' if name else None),
        DecoderBlock(ngf * 1, name=name + 'decoder_block7' if name else None),
        ReLU(name=name + 'decoder_block8/relu' if name else None),
        Conv2DTranspose(img_shape[-1], kernel_size=4, strides=2, padding='same',
                        name=name + 'decoder_block8/conv' if name else None),
        Activation('tanh', name=name + 'output_block/tanh' if name else None)
    )(img_input)

    return Model(inputs=[img_input], outputs=[output])


def unet(img_shape, ngf, name=None):

    outs = []
    layer_num = int(log2(max(img_shape[:2])))

    if name is not None:
        name = name + '/'

    img_input = Input(img_shape, name=name + 'input' if name else None)

    x = Conv2D(ngf, kernel_size=4, strides=2, padding='same', use_bias=False,
               name='encoder_block1/conv' if name else None)(img_input)
    outs.append(x)

    for i in range(1, layer_num):
        multiplier = min(2 ** i, 8)
        x = EncoderBlock(ngf * multiplier,
                         name=name + 'encoder_block{0}'.format(i + 1) if name else None)(x)
        outs.append(x)

    for i in range(layer_num - 1):
        multiplier = min(2 ** (layer_num - i - 2), 8)
        if i < 3:
            x = DecoderBlock(
                ngf * multiplier, dropout=True,
                name=name + 'decoder_block{0}'.format(i + 1) if name else None)(x)
        else:
            x = DecoderBlock(
                ngf * multiplier,
                name=name + 'decoder_block{0}'.format(i + 1) if name else None)(x)
        x = Concatenate()([x, outs[-(i + 2)]])

    output = compose(
        ReLU(name=name + 'decoder_block8/relu' if name else None),
        Conv2DTranspose(img_shape[-1], kernel_size=4, strides=2, padding='same',
                        name=name + 'decoder_block8/conv' if name else None),
        Activation('tanh', name=name + 'output_block/tanh' if name else None)
    )(x)

    return Model(inputs=[img_input], outputs=[output])


def build_conv_block(dim, padding_type, name=None):

    if name is not None:
        name = name + '/'

    if padding_type == 'reflect':
        return compose(
            ReflectionPadding2D(
                1, name=name + 'reflect_pad1' if name else None),
            Conv2D(dim, kernel_size=3, strides=1, padding='valid',
                   name=name + 'conv1' if name else None),
            BatchNormalization(name=name + 'bn1' if name else None),
            ReLU(name=name + 'relu' if name else None),
            ReflectionPadding2D(
                1, name=name + 'reflect_pad2' if name else None),
            Conv2D(dim, kernel_size=3, strides=1, padding='valid',
                   name=name + 'conv2' if name else None),
            BatchNormalization(name=name + 'bn2' if name else None)
        )

    if padding_type == 'zero':
        return compose(
            Conv2D(dim, kernel_size=3, strides=1, padding='same',
                   name=name + 'conv1' if name else None),
            BatchNormalization(name=name + 'bn1' if name else None),
            ReLU(name=name + 'relu' if name else None),
            Conv2D(dim, kernel_size=3, strides=1, padding='same',
                   name=name + 'conv2' if name else None),
            BatchNormalization(name=name + 'bn2' if name else None)
        )


def build_res_block(dim, padding_type, name=None):

    if name is not None:
        name = name + '/'

    return lambda x: Add(name=name + 'add' if name else None)([
        build_conv_block(dim, padding_type, name=name)(x),
        x
    ])


def resnet_6blocks(img_shape, ngf, name=None):

    if name is not None:
        name = name + '/'

    img_input = Input(img_shape, name=name + 'input' if name else None)

    layers = compose(
        ReflectionPadding2D(
            3, name=name + 'reflect_in/reflect_pad' if name else None),
        Conv2D(img_shape[-1], kernel_size=7, strides=1,
               padding='valid', name=name + 'reflect_in/conv' if name else None),
        BatchNormalization(name=name + 'reflect_in/relu' if name else None),
        ReLU(name=name + 'reflect_in/bn' if name else None),

        Conv2D(ngf * 2, kernel_size=3, strides=2, padding='same',
               name=name + 'ds_block1/conv' if name else None),
        BatchNormalization(name=name + 'ds_block1/bn' if name else None),
        ReLU(name=name + 'ds_block1/relu' if name else None),

        Conv2D(ngf * 4, kernel_size=3, strides=2, padding='same',
               name=name + 'ds_block2/conv' if name else None),
        BatchNormalization(name=name + 'ds_block2/bn' if name else None),
        ReLU(name=name + 'ds_block2/relu' if name else None),

        build_res_block(ngf * 4, 'reflect', name=name + 'res_block1'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block2'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block3'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block4'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block5'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block6'),

        Conv2DTranspose(ngf * 2, kernel_size=3, strides=2,
                        padding='same', name=name + 'us_block1/tconv' if name else None),
        BatchNormalization(name=name + 'us_block1/bn' if name else None),
        ReLU(name=name + 'us_block1/relu' if name else None),

        Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same',
                        name=name + 'us_block2/tconv' if name else None),
        BatchNormalization(name=name + 'us_block2/bn' if name else None),
        ReLU(name=name + 'us_block2/relu' if name else None),

        ReflectionPadding2D(
            3, name=name + 'reflect_out/reflect_pad' if name else None),
        Conv2D(img_shape[-1], kernel_size=7, strides=1,
               padding='valid', name=name + 'reflect_out/conv' if name else None),
        Activation('tanh', name=name + 'reflect_out/tanh' if name else None)
    )

    return Model(inputs=[img_input], outputs=[layers(img_input)])


def resnet_9blocks(img_shape, ngf, name=None):

    if name is not None:
        name = name + '/'

    img_input = Input(img_shape, name=name + 'input' if name else None)

    layers = compose(
        ReflectionPadding2D(
            3, name=name + 'reflect_in/reflect_pad' if name else None),
        Conv2D(img_shape[-1], kernel_size=7, strides=1,
               padding='valid', name=name + 'reflect_in/conv' if name else None),
        BatchNormalization(name=name + 'reflect_in/relu' if name else None),
        ReLU(name=name + 'reflect_in/bn' if name else None),

        Conv2D(ngf * 2, kernel_size=3, strides=2, padding='same',
               name=name + 'ds_block1/conv' if name else None),
        BatchNormalization(name=name + 'ds_block1/bn' if name else None),
        ReLU(name=name + 'ds_block1/relu' if name else None),

        Conv2D(ngf * 4, kernel_size=3, strides=2, padding='same',
               name=name + 'ds_block2/conv' if name else None),
        BatchNormalization(name=name + 'ds_block2/bn' if name else None),
        ReLU(name=name + 'ds_block2/relu' if name else None),

        build_res_block(ngf * 4, 'reflect', name=name + 'res_block1'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block2'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block3'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block4'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block5'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block6'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block7'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block8'),
        build_res_block(ngf * 4, 'reflect', name=name + 'res_block9'),

        Conv2DTranspose(ngf * 2, kernel_size=3, strides=2,
                        padding='same', name=name + 'us_block1/tconv' if name else None),
        BatchNormalization(name=name + 'us_block1/bn' if name else None),
        ReLU(name=name + 'us_block1/relu' if name else None),

        Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same',
                        name=name + 'us_block2/tconv' if name else None),
        BatchNormalization(name=name + 'us_block2/bn' if name else None),
        ReLU(name=name + 'us_block2/relu' if name else None),

        ReflectionPadding2D(
            3, name=name + 'reflect_out/reflect_pad' if name else None),
        Conv2D(img_shape[-1], kernel_size=7, strides=1,
               padding='valid', name=name + 'reflect_out/conv' if name else None),
        Activation('tanh', name=name + 'reflect_out/tanh' if name else None)
    )

    return Model(inputs=[img_input], outputs=[layers(img_input)])
