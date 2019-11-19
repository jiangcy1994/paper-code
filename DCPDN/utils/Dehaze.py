from .blocks import *
from compose import *
from .Dense import *
from .G import *
from keras import backend as K
from keras.layers import Activation, Add, AvgPool2D, Concatenate, Conv2D, Input, Lambda, LeakyReLU, Multiply, Subtract UpSampling2D
from keras.models import Model

__all__ = ['Dehaze']

def Dehaze(img_shape=(512, 512, 3)):
    img_input = Input(img_shape)
    G_model = G(img_shape=img_shape[:2])
    G2_model = G2(img_shape=img_shape[:2])
    
    tran = G_model(img_input)
    atp = G2_model(img_input)
    
#     zz = K.abs(tran) + 10**-10
    zz = Lambda(function=lambda x: 1 / (K.abs(x) + 10**-10))(tran)
    atp = AvgPool2D()(atp)
    atp = UpSampling2D()(LeakyReLU(alpha=0.2)(atp))
    
#     dehaze = (inp - atp) / zz + atp
    dehaze = Subtract()([inp, atp])
    dehaze = Multiply()([dehaze, zz])
    dehaze = Add()([dehaze, atp])

    dehaze2 = dehaze
    
    dehaze = compose(
        Concatenate(),
        Conv2D(6, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(20, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2)
    )([dehaze, img_input])
    
    
    sampling_block = lambda pool: Sampling_Block(pool, kernel_size=1)
    dehaze = compose(
        Concatenate(),
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )([
        sampling_block(32)(dehaze),
        sampling_block(16)(dehaze),
        sampling_block(8)(dehaze),
        sampling_block(4)(dehaze),
        dehaze
    ])
    
    return Model(inputs=[img_input], outputs=[dehaze, tran, atp, dehaze2])
