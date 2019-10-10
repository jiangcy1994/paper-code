from keras.layers import Activation, AvgPool2D, BatchNormalization, Concatenate, Conv2D, Input, LeakyReLU, UpSampling2D
from keras.models import Model

from .blocks import BottleneckBlock, TransitionBlock3
from .compose import compose

__all__ = ['Dense_rain_residual']

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
    
    x5 = BottleneckBlock(x4, 16, 8)
    x5 = TransitionBlock3(x5, 24, 8)
    
    x5 = Concatenate()([x5, x1])
    
    x6 = BottleneckBlock(x5, 8, 8)
    x6 = TransitionBlock3(x6, 16, 2)
    
    return x6

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
    
    x5 = BottleneckBlock(x4, 12, 8)
    x5 = TransitionBlock3(x5, 20, 4)
    
    x5 = Concatenate()([x5, x1])
    
    x6 = BottleneckBlock(x5, 4, 8)
    x6 = TransitionBlock3(x6, 12, 2)
    
    return x6

def Dense_rain_residual():
    layer_input_img = Input((128,128,3))
    # all zeros in layer_input_label
    layer_input_label = Input((128,128,4)) 

    t3 = Dense_base_down2(layer_input_img)
    t2 = Dense_base_down1(layer_input_img)
    t1 = Dense_base_down0(layer_input_img)
    t = layer_input_img

    t = Concatenate()([t1, t, t2, t3, layer_input_label])
    t = compose(
        Conv2D(20, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(alpha=0.2)
    )(t)

    t101 = AvgPool2D(32)(t)
    t102 = AvgPool2D(16)(t)
    t103 = AvgPool2D(8)(t)
    t104 = AvgPool2D(4)(t)

    t1010 = compose(
        Conv2D(1, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(32)
    )(t101)
    t1020 = compose(
        Conv2D(1, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(16)
    )(t102)
    t1030 = compose(
        Conv2D(1, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(8)
    )(t103)
    t1040 = compose(
        Conv2D(1, kernel_size=1, strides=1, padding='same'),
        LeakyReLU(alpha=0.2),
        UpSampling2D(4)
    )(t104)

    t = Concatenate()([t1010, t1020, t1030, t1040, t])

    out = compose(
        Conv2D(3, kernel_size=3, strides=1, padding='same'),
        Activation('tanh')
    )(t)

    model = Model(inputs=[layer_input_img, layer_input_label], outputs=[out])
    return model
