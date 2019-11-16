from .blocks import *
from compose import *
from keras.layers import Activation, Concatenate, Conv2D, Input, LeakyReLU
from keras.models import Model

__all__ = ['Dense_rain_residual']

def Dense1(x, prefix=None):
    '''
    2 transition-down layers
    2 no-sampling transition layers
    2 transition-up layers
    kernel size (3, 3)
    '''
    if not prefix:
        prefix = ''
    
    # transition-down layer 1
    x1 = BottleneckBlock_3(13, name=prefix + 'dense1/db_1/bn')(x)
    x1 = TransitionBlock_Down(8, name=prefix + 'dense1/db_1/td')(x1)
    
    # transition-down layer 2
    x2 = BottleneckBlock_3(16, name=prefix + 'dense1/db_2/bn')(x1)
    x2 = TransitionBlock_Down(16, name=prefix + 'dense1/db_2/td')(x2)
    
    # no-sampling transition layer 1
    x3 = BottleneckBlock_3(16, name=prefix + 'dense1/db_3/bn')(x2)
    x3 = TransitionBlock_Plain(16, name=prefix + 'dense1/db_3/tp')(x3)
    
    # no-sampling transition layer 2
    x4 = BottleneckBlock_3(16, name=prefix + 'dense1/db_4/bn')(x3)
    x4 = TransitionBlock_Plain(16, name=prefix + 'dense1/db_4/tp')(x4)
    x4 = Concatenate(name=prefix + 'dense1/concat1')([x4, x2])
    
    # transition-up layer 1
    x5 = BottleneckBlock_3(16, name=prefix + 'dense1/db_5/bn')(x4)
    x5 = TransitionBlock_Up(16, name=prefix + 'dense1/db_5/tu')(x5)
    x5 = Concatenate(name=prefix + 'dense1/concat2')([x5, x1])
    
    # transition-up layer 2
    x6 = BottleneckBlock_3(8, name=prefix + 'dense1/db_6/bn')(x5)
    x6 = TransitionBlock_Up(2, name=prefix + 'dense1/db_6/tu')(x6)
    
    return x6

def Dense2(x, prefix=None):
    '''
    1 transition-down layers
    4 no-sampling transition layers
    1 transition-up layers
    kernel size (3, 3)
    '''
    if not prefix:
        prefix = ''
    
    # transition-down layer 1
    x1 = BottleneckBlock_3(13, name=prefix + 'dense2/db_1/bn')(x)
    x1 = TransitionBlock_Down(8, name=prefix + 'dense2/db_1/td')(x1)
    
    # no-sampling transition layer 1
    x2 = BottleneckBlock_3(16, name=prefix + 'dense2/db_2/bn')(x1)
    x2 = TransitionBlock_Plain(16, name=prefix + 'dense2/db_2/td')(x2)
    
    # no-sampling transition layer 2
    x3 = BottleneckBlock_3(16, name=prefix + 'dense2/db_3/bn')(x2)
    x3 = TransitionBlock_Plain(16, name=prefix + 'dense2/db_3/tp')(x3)
    
    # no-sampling transition layer 3
    x4 = BottleneckBlock_3(16, name=prefix + 'dense2/db_4/bn')(x3)
    x4 = TransitionBlock_Plain(16, name=prefix + 'dense2/db_4/tp')(x4)
    x4 = Concatenate(name=prefix + 'dense2/concat1')([x4, x2])
    
    # no-sampling transition layer 4
    x5 = BottleneckBlock_3(16, name=prefix + 'dense2/db_5/bn')(x4)
    x5 = TransitionBlock_Plain(16, name=prefix + 'dense2/db_5/tu')(x5)
    x5 = Concatenate(name=prefix + 'dense2/concat2')([x5, x1])
    
    # transition-up layer 1
    x6 = BottleneckBlock_3(8, name=prefix + 'dense2/db_6/bn')(x5)
    x6 = TransitionBlock_Up(2, name=prefix + 'dense2/db_6/tu')(x6)
    
    return x6

def Dense3(x, prefix=None):
    '''
    6 no-sampling transition layers
    kernel size (3, 3)
    '''
    if not prefix:
        prefix = ''
    
    # no-sampling transition layer 1
    x1 = BottleneckBlock_3(5, name=prefix + 'dense3/db_1/bn')(x)
    x1 = TransitionBlock_Plain(4, name=prefix + 'dense3/db_1/td')(x1)
    
    # no-sampling transition layer 2
    x2 = BottleneckBlock_3(8, name=prefix + 'dense3/db_2/bn')(x1)
    x2 = TransitionBlock_Plain(12, name=prefix + 'dense3/db_2/td')(x2)
    
    # no-sampling transition layer 3
    x3 = BottleneckBlock_3(4, name=prefix + 'dense3/db_3/bn')(x2)
    x3 = TransitionBlock_Plain(12, name=prefix + 'dense3/db_3/tp')(x3)
    
    # no-sampling transition layer 4
    x4 = BottleneckBlock_3(4, name=prefix + 'dense3/db_4/bn')(x3)
    x4 = TransitionBlock_Plain(12, name=prefix + 'dense3/db_4/tp')(x4)
    x4 = Concatenate(name=prefix + 'dense3/concat1')([x4, x2])
    
    # no-sampling transition layer 5
    x5 = BottleneckBlock_3(8, name=prefix + 'dense3/db_5/bn')(x4)
    x5 = TransitionBlock_Plain(4, name=prefix + 'dense3/db_5/tu')(x5)
    x5 = Concatenate(name=prefix + 'dense3/concat2')([x5, x1])
    
    # no-sampling transition layer 6
    x6 = BottleneckBlock_3(8, name=prefix + 'dense3/db_6/bn')(x5)
    x6 = TransitionBlock_Plain(2, name=prefix + 'dense3/db_6/tu')(x6)
    
    return x6

def Dense_rain_residual(img_shape=(128,128,3)):
    '''
    Residual-aware Rain-density Classifier
    '''
    
    img_input = Input(img_shape, name='img_input')
    label_input = Input(img_shape[:2] + (4,), name='label_input') 
    prefix='rainclass/'

    x = img_input
    x1 = Dense1(x, prefix=prefix)
    x2 = Dense2(x, prefix=prefix)
    x3 = Dense3(x, prefix=prefix)

    x = compose(
        Concatenate(name=prefix + 'concat1'),
        Conv2D(20, kernel_size=3, strides=1, padding='same', name=prefix + 'concat1/conv2d'),
        LeakyReLU(alpha=0.2, name=prefix + 'concat1/lrelu')
    )([x3, x, x2, x1, label_input])
    
    x101 = Sampling_Block(32, name=prefix + 'sample32')(x)
    x102 = Sampling_Block(16, name=prefix + 'sample16')(x)
    x103 = Sampling_Block(8, name=prefix + 'sample8')(x)
    x104 = Sampling_Block(4, name=prefix + 'sample4')(x)
    x = Concatenate(name=prefix + 'concat2')([x101, x102, x103, x104, x])

    out = compose(
        Conv2D(3, kernel_size=3, strides=1, padding='same', name=prefix + 'out/conv2d'),
        Activation('tanh', name=prefix + 'out/tanh')
    )(x)

    model = Model(inputs=[img_input, label_input], outputs=[out])
    return model
