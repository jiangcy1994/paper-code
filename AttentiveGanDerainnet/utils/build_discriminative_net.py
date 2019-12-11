from compose import *
from tensorflow.keras.layers import Activation, Conv2D, Dense, Input, LeakyReLU, Multiply
from tensorflow.keras.models import Model

__all__ = ['build_discriminative_net']

conv_stage = lambda filters, kernal_size, strides, name: compose(
        Conv2D(filters, kernel_size=kernal_size, strides=strides, padding='same',
               use_bias=False, name=name + '/conv'),
        LeakyReLU(name=name + '/lrelu')
    )

def build_discriminative_net(img_shape):
    
    img_input = Input(img_shape)
    
    conv_stage_6 = compose(
        conv_stage(8, 5, 1, 'conv_stage_1'),
        conv_stage(16, 5, 1, 'conv_stage_2'),
        conv_stage(32, 5, 1, 'conv_stage_3'),
        conv_stage(64, 5, 1, 'conv_stage_4'),
        conv_stage(128, 5, 1, 'conv_stage_5'),
        conv_stage(128, 5, 1, 'conv_stage_6')
    )(img_input)
    
    attention_map = Conv2D(1, kernel_size=5, strides=1, padding='same', use_bias=False, 
                           name='attention_map')(conv_stage_6)

    fc_2 = compose(
        Multiply(),
        conv_stage(64, 5, 4, 'conv_stage_7'),
        conv_stage(64, 5, 4, 'conv_stage_8'),
        conv_stage(32, 5, 4, 'conv_stage_9'),
        Dense(1024, use_bias=False, name='fc_1'),
        Dense(1, use_bias=False, name='fc_2')
    )([attention_map, conv_stage_6])
    
    fc_out = compose(
        Activation('sigmoid', name='fc_out'),
#         tf.where(tf.not_equal(fc_out, 1.0), fc_out, fc_out - 0.0000001) ???
#         tf.where(tf.not_equal(fc_out, 0.0), fc_out, fc_out + 0.0000001) ???
    )(fc_2)
    
    return Model(inputs=[img_input], outputs=[fc_out, attention_map, fc_2])
