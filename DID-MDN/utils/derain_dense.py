from .blocks import *
from compose import *
from keras.layers import Activation, Concatenate, Conv2D, Input, LeakyReLU, Subtract
from keras.models import Model

__all__ = ['Dense_rain']

def Dense1(x):
    '''
    3 transition-down layers
    3 transition-up layers 
    kernel size (7, 7)
    '''
    
    # transition-down layer 1
    x1 = BottleneckBlock_7(13, name='dense1/db_1/bn')(x)
    x1 = TransitionBlock_Down(8, name='dense1/db_1/td')(x1)
    x1_final = UpSampling_Block(2, name='dense1/us_1')(x1)
    
    # transition-down layer 2
    x2 = BottleneckBlock_7(16, name='dense1/db_2/bn')(x1)
    x2 = TransitionBlock_Down(16, name='dense1/db_2/td')(x2)
    x2_final = UpSampling_Block(4, name='dense1/us_2')(x2)
    
    # transition-down layer 3
    x3 = BottleneckBlock_7(16, name='dense1/db_3/bn')(x2)
    x3 = TransitionBlock_Down(16, name='dense1/db_3/td')(x3)
    x3_final = UpSampling_Block(8, name='dense1/us_3')(x3)
    
    # transition-up layer 1
    x4 = BottleneckBlock_7(16, name='dense1/db_4/bn')(x3)
    x4 = TransitionBlock_Up(16, name='dense1/db_4/tu')(x4)
    x4 = Concatenate(name='dense1/concat1')([x4, x2])
    x4_final = UpSampling_Block(4, name='dense1/us_4')(x4)
    
    # transition-up layer 2
    x5 = BottleneckBlock_7(16, name='dense1/db_5/bn')(x4)
    x5 = TransitionBlock_Up(16, name='dense1/db_5/tu')(x5)
    x5 = Concatenate(name='dense1/concat2')([x5, x1])
    x5_final = UpSampling_Block(2, name='dense1/us_5')(x5)
    
    # transition-up layer 3
    x6 = BottleneckBlock_7(8, name='dense1/db_6/bn')(x5)
    x6_final = TransitionBlock_Up(4, name='dense1/db_6/tu')(x6)
    
    # concat all
    out = Concatenate(name='dense1/concat3')(
        [x6_final, x5_final, x4_final, x3_final, x2_final, x1_final, x]
    )
    
    return out

def Dense2(x):
    '''
    2 transition-down layers
    2 no-sampling transition layers
    2 transition-up layers
    kernel size (5, 5)
    '''
    
    # transition-down layer 1
    x1 = BottleneckBlock_5(13, name='dense2/db_1/bn')(x)
    x1 = TransitionBlock_Down(8, name='dense2/db_1/td')(x1)
    x1_final = UpSampling_Block(2, name='dense2/us_1')(x1)
    
    # transition-down layer 2
    x2 = BottleneckBlock_5(16, name='dense2/db_2/bn')(x1)
    x2 = TransitionBlock_Down(16, name='dense2/db_2/td')(x2)
    x2_final = UpSampling_Block(4, name='dense2/us_2')(x2)
    
    # no-sampling transition layer 1
    x3 = BottleneckBlock_5(16, name='dense2/db_3/bn')(x2)
    x3 = TransitionBlock_Plain(16, name='dense2/db_3/tp')(x3)
    x3_final = UpSampling_Block(4, name='dense2/us_3')(x3)
    
    # no-sampling transition layer 2
    x4 = BottleneckBlock_5(16, name='dense2/db_4/bn')(x3)
    x4 = TransitionBlock_Plain(16, name='dense2/db_4/tp')(x4)
    x4 = Concatenate(name='dense2/concat1')([x4, x2])
    x4_final = UpSampling_Block(4, name='dense2/us_4')(x4)
    
    # transition-up layer 1
    x5 = BottleneckBlock_5(16, name='dense2/db_5/bn')(x4)
    x5 = TransitionBlock_Up(16, name='dense2/db_5/tu')(x5)
    x5 = Concatenate(name='dense2/concat2')([x5, x1])
    x5_final = UpSampling_Block(2, name='dense2/us_5')(x5)
    
    # transition-up layer 2
    x6 = BottleneckBlock_5(8, name='dense2/db_6/bn')(x5)
    x6_final = TransitionBlock_Up(4, name='dense2/db_6/tu')(x6)
    
    # concat all
    out = Concatenate(name='dense2/concat3')([x6_final, x5_final, x4_final, x3_final, x2_final, x1_final, x])
    
    return out

def Dense3(x):
    '''
    1 transition-down layers
    4 no-sampling transition layers
    1 transition-up layers
    kernel size (5, 5)
    '''
    
    # transition-down layer 1
    x1 = BottleneckBlock_5(5, name='dense3/db_1/bn')(x)
    x1 = TransitionBlock_Down(4, name='dense3/db_1/td')(x1)
    x1_final = UpSampling_Block(2, name='dense3/us_1')(x1)
    
    # no-sampling transition layer 1
    x2 = BottleneckBlock_5(8, name='dense3/db_2/bn')(x1)
    x2 = TransitionBlock_Plain(12, name='dense3/db_2/td')(x2)
    x2_final = UpSampling_Block(2, name='dense3/us_2')(x2)
    
    # no-sampling transition layer 2
    x3 = BottleneckBlock_5(4, name='dense3/db_3/bn')(x2)
    x3 = TransitionBlock_Plain(12, name='dense3/db_3/tp')(x3)
    x3_final = UpSampling_Block(2, name='dense3/us_3')(x3)
    
    # no-sampling transition layer 3
    x4 = BottleneckBlock_5(4, name='dense3/db_4/bn')(x3)
    x4 = TransitionBlock_Plain(12, name='dense3/db_4/tp')(x4)
    x4 = Concatenate(name='dense3/concat1')([x4, x2])
    x4_final = UpSampling_Block(2, name='dense3/us_4')(x4)
    
    # no-sampling transition layer 4
    x5 = BottleneckBlock_5(8, name='dense3/db_5/bn')(x4)
    x5 = TransitionBlock_Plain(4, name='dense3/db_5/tu')(x5)
    x5 = Concatenate(name='dense3/concat2')([x5, x1])
    x5_final = UpSampling_Block(2, name='dense3/us_5')(x5)
    
    # transition-up layer 1
    x6 = BottleneckBlock_5(8, name='dense3/db_6/bn')(x5)
    x6_final = TransitionBlock_Up(4, name='dense3/db_6/tu')(x6)
    
    # concat all
    out = Concatenate(name='dense3/concat3')([x6_final, x5_final, x4_final, x3_final, x2_final, x1_final, x])
    
    return out

def Dense_rain(img_shape=(128, 128, 3)):
    '''
    Multi-stream Dense Network
    '''
    
    img_input = Input(img_shape, name='img_input')
    label_input = Input(img_shape[:2] + (8,), name='label_input') 

    x = img_input
    x1 = Dense1(x)
    x2 = Dense2(x)
    x3 = Dense3(x)

    x = Concatenate(name='concat1')([x3, x, x2, x1, label_input])
    x = compose(
        Conv2D(47, kernel_size=3, strides=1, padding='same', name='concat1/conv2d'),
        LeakyReLU(alpha=0.2, name='concat1/lrelu')
    )(x)
    
    x101 = Sampling_Block(32, name='sample32')(x)
    x102 = Sampling_Block(16, name='sample16')(x)
    x103 = Sampling_Block(8, name='sample8')(x)
    x104 = Sampling_Block(4, name='sample4')(x)
    x = Concatenate(name='concat2')([x101, x102, x103, x104, x])

    residual = compose(
        Conv2D(3, kernel_size=3, strides=1, padding='same', name='residual/conv2d'),
        Activation('tanh', name='residual/tanh')
    )(x)
    
    clear = compose(
        Subtract(name='clear/subtract'),
        Conv2D(8, kernel_size=7, strides=1, padding='same', name='clear/conv2d1'),
        LeakyReLU(alpha=0.2, name='clear/lrelu'),
        Conv2D(3, kernel_size=3, strides=1, padding='same', name='clear/conv2d2'),
        Activation('tanh', name='clear/tanh')
    )([img_input, residual])
    
    model = Model(inputs=[img_input, label_input], outputs=[residual, clear])
    return model
