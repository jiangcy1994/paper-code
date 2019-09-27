from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout, Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, ReLU, LeakyReLU, Concatenate
from functools import partial

__all__ = ['BASIC_D', 'UNET_G']

batchnorm = partial(BatchNormalization, momentum=0.9, epsilon=1.01e-5)

# Basic discriminator
def BASIC_D(num_channels_in, num_discriminator_filter, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       num_channels_in: channels
       num_discriminator_filter: filters of the first layer
       max_layers: max hidden layers
    """    
    input_a = Input(shape=(None, None, num_channels_in))
    t = input_a
    t = Conv2D(num_discriminator_filter, kernel_size=4, strides=2, padding="same", name='First') (t)
    t = LeakyReLU(alpha=0.2)(t)

    for layer in range(1, max_layers):
        out_feat = num_discriminator_filter * min(2**layer, 8)
        t = Conv2D(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name='pyramid.{0}'.format(layer)
                  )(t)
        t = batchnorm()(t, training=1)
        t = LeakyReLU(alpha=0.2)(t)

    out_feat = num_discriminator_filter*min(2**max_layers, 8)
    t = ZeroPadding2D()(t)
    t = Conv2D(out_feat, kernel_size=4,  use_bias=False, name='pyramid_last')(t)
    t = batchnorm()(t, training=1)
    t = LeakyReLU(alpha=0.2)(t)

    # final layer
    t = ZeroPadding2D()(t)
    t = Conv2D(1, kernel_size=4, name='final'.format(out_feat, 1), 
               activation="sigmoid" if use_sigmoid else None
              )(t)    
    return Model(inputs=[input_a], outputs=t)
    
    
def UNET_G(isize, num_channel_in=3, num_channel_out=3, num_generator_filter=64, fixed_input_size=True):
    """U-Net Generator"""
    max_num_filter = 8 * num_generator_filter
    
    def block(x, size, num_filter_in, use_batchnorm=True, num_filter_out=None, num_filter_next=None):
        assert size >= 2 and size % 2 == 0
        if num_filter_next is None:
            num_filter_next = min(num_filter_in*2, max_num_filter)
        if num_filter_out is None:
            num_filter_out = num_filter_in
        x = Conv2D(num_filter_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and size > 2)),
                   padding='same', name='conv_{0}'.format(size)
                  )(x)
        if size > 2:
            if use_batchnorm:
                x = BatchNormalization(momentum=0.9, epsilon=1.01e-5)(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, size//2, num_filter_next)
            x = Concatenate()([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(num_filter_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            name='convt.{0}'.format(size))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = BatchNormalization(momentum=0.9, epsilon=1.01e-5)(x, training=1)
        if size <= 8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    size = isize if fixed_input_size else None
    t = inputs = Input(shape=(size, size, num_channel_in))        
    t = block(t, isize, num_channel_in, False, num_filter_out=num_channel_out, num_filter_next=num_generator_filter)
    t = Activation('tanh')(t)
    return Model(inputs=inputs, outputs=[t])
