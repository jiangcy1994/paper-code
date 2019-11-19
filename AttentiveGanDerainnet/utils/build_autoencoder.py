from compose import *
from functools import partial
from keras.layers import Activation, AvgPool2D, Conv2D, Deconv2D, Input, LeakyReLU
from keras.models import Model

__all__ = ['build_autoencoder']

def build_autoencoder(img_shape):
    img_input = Input(img_shape[:-1] + (4,))
    
    block_conv_lrelu = lambda filters, kernal_size, dilation_rate=1: compose(
        Conv2D(filters, kernel_size=kernal_size, strides=1, padding='same', 
               dilation_rate=dilation_rate, use_bias=False),
        LeakyReLU()
    )
    
    block_deconv_avgpool_lrelu = lambda filters: compose(
        Deconv2D(filters, kernel_size=4, strides=2, padding='same', use_bias=False),
        AvgPool2D(),
        LeakyReLU()
    )
    
    skip_conv = partial(Conv2D, filters=3, kernel_size=3, strides=1, padding='same', use_bias=False)
    
    # conv1 -> relu12
    relu_12 = compose(
        block_conv_lrelu(64, 5),
        block_conv_lrelu(128, 3),
        block_conv_lrelu(128, 3),
        block_conv_lrelu(128, 3),
        block_conv_lrelu(256, 3),
        block_conv_lrelu(256, 3),
        block_conv_lrelu(256, 3, 2),
        block_conv_lrelu(256, 3, 4),
        block_conv_lrelu(256, 3, 8),
        block_conv_lrelu(256, 3, 16),
        block_conv_lrelu(256, 3),
        block_conv_lrelu(256, 3)
    )(img_input)
    
    relu_14 = compose(
        block_deconv_avgpool_lrelu(128),
        block_conv_lrelu(128, 3)
    )(relu_12)
    
    relu_16 = compose(
        block_deconv_avgpool_lrelu(64),
        block_conv_lrelu(32, 3)
    )(relu_14)
    
    skip_output_1 = skip_conv()(relu_12)
    skip_output_2 = skip_conv()(relu_14)
    skip_output_3 = skip_conv()(relu_16)
    skip_output_3 = Activation('tanh')(skip_output_3)
    
    return Model(inputs=[img_input],
                 outputs=[skip_output_1, skip_output_2, skip_output_3])
