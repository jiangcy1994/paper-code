from compose import *
from keras.layers import Add, Conv2D, LeakyReLU

__all__ = ['residual_block']

block1 = lambda name: compose(
    Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name=name + 'block_1/conv2d'),
    LeakyReLU(name=name + 'block1/relu_1')
)
    
blockn_1 = lambda n, name: compose(
    Conv2D(32, kernel_size=1, strides=1, padding='same', use_bias=False, name=name + 'block_{0}/conv2d_1'.format(n)),
    LeakyReLU(name=name + 'block_{0}/relu_1'.format(n)),
    Conv2D(32, kernel_size=1, strides=1, padding='same', use_bias=False, name=name + 'block_{0}/conv2d_2'.format(n)),
    LeakyReLU(name=name + 'block_{0}/relu_2'.format(n))
)

blockn_2 = lambda n, name: compose(
    Add(name=name + 'block_{0}/add'.format(n)),
    LeakyReLU(name=name + 'block_{0}/relu_3'.format(n))
)

def residual_block(input_tensor, name=None):
    
    if type(name) is str:
        name = name + '/'
    
    shortcut = inputs = input_tensor
    
    inputs = block1(name)(inputs)
    shortcut = output = inputs
    
    for i in range(1, 5):
        inputs = blockn_1(i + 1, name)(inputs)
        output = blockn_2(i + 1, name)([inputs, shortcut])
        shortcut = output
    
    return output
