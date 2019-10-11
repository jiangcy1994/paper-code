from keras.layers import Activation, Concatenate, Conv2D, Input
from keras.models import Model

from .blocks import UNetBlock, Sampling_Block

__all__ = ['G', 'G2']

def G(input_nc=3, output_nc=3, nf=64):
    din = Input((256, 256, input_nc))
    x = din
    outs = []
    
    x = Conv2D(nf, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')(x)
    outs.append(x)
    
    for layer_idx, nf_multi in enumerate([2, 4, 8, 8, 8, 8, 8]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock(x, nf * nf_multi, name, transposed=False, bn=True, relu=False, dropout=False)
        outs.append(x)

    x = UNetBlock(x, nf*8, 'dlayer%d' % 8, transposed=True, bn=False, relu=True, dropout=True)

    for layer_idx, nf_multi in enumerate([8, 8, 8, 4, 2, 1]):
        layer_idx = 7 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = Concatenate()([x, outs[layer_idx - 1]])
        x = UNetBlock(x, nf * nf_multi, name, transposed=True, bn=True, relu=True, dropout=False)

    x = Concatenate()([x, outs[0]])
    x = UNetBlock(x, 20, 'dlayer%d' % 1, transposed=True, bn=False, relu=True, dropout=False)
    
    x1010 = Sampling_Block(16)(x)    
    x1020 = Sampling_Block(8)(x)    
    x1030 = Sampling_Block(4)(x)    
    x1040 = Sampling_Block(2)(x)

    dehaze = Concatenate()([x1010, x1020, x1030, x1040, x])
    dout = Conv2D(output_nc, kernel_size=3, strides=1, padding='same', use_bias=False, name='layerfinal.conv')(dehaze)
    dout = Activation('tanh', name='layerfinal.tanh')(dout)

    model = Model(inputs=[din], outputs=[dout])
    return model

def G2(input_nc=3, output_nc=3, nf=8):
    din = Input((256, 256, input_nc))
    x = din
    outs = []
    
    x = Conv2D(nf, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')(x)
    outs.append(x)
    
    for layer_idx, nf_multi in enumerate([2, 4, 8, 8, 8, 8, 8]):
        name = 'layer%d' % (layer_idx + 2)
        x = UNetBlock(x, nf * nf_multi, name, transposed=False, bn=True, relu=False, dropout=False)
        outs.append(x)

    x = UNetBlock(x, nf*8, 'dlayer8', transposed=True, bn=False, relu=True, dropout=True)

    for layer_idx, nf_multi in enumerate([8, 8, 8, 4, 2, 1]):
        layer_idx = 7 - layer_idx
        name = 'dlayer%d' % layer_idx
        x = Concatenate()([x, outs[layer_idx - 1]])
        x = UNetBlock(x, nf * nf_multi, name, transposed=True, bn=True, relu=True, dropout=False)

    x = Concatenate()([x, outs[0]])
    x = UNetBlock(x, output_nc, 'dlayer1', transposed=True, bn=False, relu=True, dropout=False)
    x = Activation('tanh', name='dlayer1.tanh')(x)

    model = Model(inputs=[din], outputs=[x])
    return model
