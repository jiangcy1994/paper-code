from keras import backend as K
from keras.layers import Activation, AvgPool2D, Concatenate, Conv2D, Input, Lambda, LeakyReLU, UpSampling2D
from keras.layers import Add, Multiply, Subtract
from keras.models import Model

from .blocks import Sampling_Block
from .G import *
from .Dense import *

__all__ = ['Dehaze']

def Dehaze():
    inp = Input((512, 512, 3))
    G_model = G()
    G2_model = G2()
    
    tran = G_model(inp)
    atp = G2_model(inp)
    
#     zz = K.abs(tran) + 10**-10
    zz = Lambda(function=lambda x: 1 / (K.abs(x) + 10**-10))(tran)
    atp = AvgPool2D()(atp)
    atp = UpSampling2D()(LeakyReLU(alpha=0.2)(atp))
    
#     dehaze = (inp - atp) / zz + atp
    dehaze = Subtract()([inp, atp])
    dehaze = Multiply()([dehaze, zz])
    dehaze = Add()([dehaze, atp])

    dehaze2 = dehaze
    
    dehaze = Concatenate()([dehaze, inp])
    dehaze = LeakyReLU(alpha=0.2)(Conv2D(20, kernel_size=3, strides=1, padding='same')(dehaze))
    dehaze = LeakyReLU(alpha=0.2)(Conv2D(20, kernel_size=3, strides=1, padding='same')(dehaze))
    
    x1010 = Sampling_Block(32, kernel_size=1)(dehaze)
    x1020 = Sampling_Block(16, kernel_size=1)(dehaze)
    x1030 = Sampling_Block(8, kernel_size=1)(dehaze)
    x1040 = Sampling_Block(4, kernel_size=1)(dehaze)
    dehaze = Concatenate()([x1010, x1020, x1030, x1040, dehaze])
    dehaze = Activation('tanh')(Conv2D(3, kernel_size=3, strides=1, padding='same')(dehaze))
    
    return Model(inputs=[inp], outputs=[dehaze, tran, atp, dehaze2])
    
    