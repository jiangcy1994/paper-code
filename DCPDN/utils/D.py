from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Input, LeakyReLU, ZeroPadding2D
from keras.models import Model

from .blocks import BottleneckBlock, TransitionBlock, UNetBlock, Sampling_Block

__all__ = ['D']

def D(nc=6, nf=64):
    
    din = Input((256, 256, nc))
    
    x = Conv2D(nf, kernel_size=4, strides=2, padding='same', use_bias=False, name='layer1')(din)
    x = UNetBlock(x, nf * 2, name='layer2', transposed=False, bn=True, relu=False, dropout=False)
    x = UNetBlock(x, nf * 4, name='layer3', transposed=False, bn=True, relu=False, dropout=False)
    
    x = LeakyReLU(alpha=0.2, name='layer4.leakyrelu')(x)
    x = Conv2D(nf * 8, kernel_size=4, strides=1, padding='valid', use_bias=False, name='layer4.conv')(x)
    x = ZeroPadding2D(name='layer4.zeropad')(x)
    x = BatchNormalization(name='layer4.bn')(x)
    
    x = LeakyReLU(alpha=0.2, name='layer5.leakyrelu')(x)
    x = Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False, name='layer5.conv')(x)
    x = ZeroPadding2D(name='layer5.zeropad')(x)
    x = Activation('sigmoid', name='layer5.sigmoid')(x)
    
    return Model(inputs=[din], outputs=[x])
