from keras.layers import Activation, AvgPool2D, BatchNormalization, Concatenate, Conv2D, Input, LeakyReLU, Subtract, UpSampling2D
from keras.models import Model

from .blocks import BottleneckBlock, Dense_Block, TransitionBlock3
from .compose import compose

__all__ = ['Dense_rain']

def Dense_base_down2(x):
    ## 256x256
    x1 = BottleneckBlock(x, 3, 13)
    x1 = TransitionBlock3(x1, 16, 8)
    
    ###  32x32
    x2 = BottleneckBlock(x1, 8, 16)
    x2 = TransitionBlock3(x2, 24, 16)
    
    ### 16 X 16
    x3 = BottleneckBlock(x2, 16, 16)
    x3 = TransitionBlock3(x3, 32, 16)
    
    ## Classifier  ##
    x4 = BottleneckBlock(x3, 16, 16)
    x4 = TransitionBlock3(x4, 32, 16)
    
    x4 = Concatenate()([x4, x2])
    
    x5 = BottleneckBlock(x4, 32, 8)
    x5 = TransitionBlock3(x5, 40, 8)
    
    x5 = Concatenate()([x5, x1])
    
    x6 = BottleneckBlock(x5, 16, 8)
    x6 = TransitionBlock3(x6, 24, 4)
    
    x11 = Dense_Block()(x1)
    x21 = Dense_Block()(x2)
    x31 = Dense_Block()(x3)
    x41 = Dense_Block()(x4)
    x51 = Dense_Block()(x5)
    
    out = Concatenate()([x6,x51,x41,x31,x21,x11,x])
    
    return out

Dense_base_down1 = Dense_base_down2

def Dense_base_down0(x):
    ## 256x256
    x1 = BottleneckBlock(x, 3, 5)
    x1 = TransitionBlock3(x1, 8, 4)
    
    ###  32x32
    x2 = BottleneckBlock(x1, 4, 8)
    x2 = TransitionBlock3(x2, 12, 12)
    
    ### 16 X 16
    x3 = BottleneckBlock(x2, 12, 4)
    x3 = TransitionBlock3(x3, 16, 12)
    
    ## Classifier  ##
    x4 = BottleneckBlock(x3, 12, 4)
    x4 = TransitionBlock3(x4, 16, 12)
    
    x4 = Concatenate()([x4, x2])
    
    x5 = BottleneckBlock(x4, 24, 8)
    x5 = TransitionBlock3(x5, 32, 4)
    
    x5 = Concatenate()([x5, x1])
    
    x6 = BottleneckBlock(x5, 8, 8)
    x6 = TransitionBlock3(x6, 16, 4)
    
    x11 = Dense_Block()(x1)
    x21 = Dense_Block()(x2)
    x31 = Dense_Block()(x3)
    x41 = Dense_Block()(x4)
    x51 = Dense_Block()(x5)
    
    out = Concatenate()([x6,x51,x41,x31,x21,x11,x])
    
    return out

def Dense_rain(img_shape=(128, 128, 3)):
    layer_input_img = Input(img_shape)
    # all ones in layer_input_label
    layer_input_label = Input(img_shape[:2] + (8,)) 

    t3 = Dense_base_down2(layer_input_img)
    t2 = Dense_base_down1(layer_input_img)
    t1 = Dense_base_down0(layer_input_img)
    t = layer_input_img

    t = Concatenate()([t1, t, t2, t3, layer_input_label])
    t = compose(
        Conv2D(47, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2)
    )(t)

    t101 = AvgPool2D(32)(t)
    t102 = AvgPool2D(16)(t)
    t103 = AvgPool2D(8)(t)
    t104 = AvgPool2D(4)(t)

    t1010 = compose(
        Conv2D(2, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(32)
    )(t101)
    t1020 = compose(
        Conv2D(2, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(16)
    )(t102)
    t1030 = compose(
        Conv2D(2, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(8)
    )(t103)
    t1040 = compose(
        Conv2D(2, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(4)
    )(t104)

    t = Concatenate()([t1010, t1020, t1030, t1040, t])

    residual = compose(
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )(t)
    
    clear = Subtract()([layer_input_img, residual])
    clear1 = compose(
        Conv2D(8, kernel_size=7, strides=1, padding='same'),
        LeakyReLU(alpha=0.2)
    )(clear)
    clear2 = compose(
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )(clear1)
    
    model = Model(inputs=[layer_input_img, layer_input_label], outputs=[residual, clear2])
    return model
