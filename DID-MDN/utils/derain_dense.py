from .blocks import *
from compose import *
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Input, LeakyReLU, Subtract
from tensorflow.keras.models import Model

__all__ = ['Dense_rain']


def _Dense(x, db_blocks, us_blocks, name=None):
    '''
    Dense Block Template
    '''
    if name is not None:
        name = name + '/'

    x1 = db_blocks[0](x)
    x2 = db_blocks[1](x1)
    x3 = db_blocks[2](x2)
    x4 = db_blocks[3](x3)
    x4 = Concatenate(name=name + 'concat1' if name else None)([x4, x2])
    x5 = db_blocks[4](x4)
    x5 = Concatenate(name=name + 'concat2' if name else None)([x5, x1])
    x6 = db_blocks[5](x5)

    out = Concatenate(name=name + 'concat3' if name else None)(
        [x6,
         us_blocks[4](x5),
         us_blocks[3](x4),
         us_blocks[2](x3),
         us_blocks[1](x2),
         us_blocks[0](x1),
         x]
    )

    return out


def Dense1(x, name='dense1'):
    '''
    3 transition-down layers
    3 transition-up layers 
    kernel size (7, 7)
    '''

    _bn_block = BottleneckBlock_7
    _bn_filters = [13, 16, 16, 16, 16, 8]
    _tr_blocks = [TransitionBlock_Down,
                  TransitionBlock_Down,
                  TransitionBlock_Down,
                  TransitionBlock_Up,
                  TransitionBlock_Up,
                  TransitionBlock_Up]
    _tr_filters = [8, 16, 16, 16, 16, 4]
    _us_pools = [2, 4, 8, 4, 2]

    db_blocks = DB_Blocks_Gen(_bn_block, _bn_filters,
                              _tr_blocks, _tr_filters, name)
    us_blocks = US_Blocks_Gen(_us_pools, name)
    return _Dense(x, db_blocks, us_blocks, name=name)


def Dense2(x, name='dense2'):
    '''
    2 transition-down layers
    2 no-sampling transition layers
    2 transition-up layers
    kernel size (5, 5)
    '''

    _bn_block = BottleneckBlock_5
    _bn_filters = [13, 16, 16, 16, 16, 8]
    _tr_blocks = [TransitionBlock_Down,
                  TransitionBlock_Down,
                  TransitionBlock_Plain,
                  TransitionBlock_Plain,
                  TransitionBlock_Up,
                  TransitionBlock_Up]
    _tr_filters = [8, 16, 16, 16, 16, 4]
    _us_pools = [2, 4, 4, 4, 2]

    db_blocks = DB_Blocks_Gen(_bn_block, _bn_filters,
                              _tr_blocks, _tr_filters, name)
    us_blocks = US_Blocks_Gen(_us_pools, name)
    return _Dense(x, db_blocks, us_blocks, name=name)


def Dense3(x, name='dense3'):
    '''
    1 transition-down layers
    4 no-sampling transition layers
    1 transition-up layers
    kernel size (5, 5)
    '''

    _bn_block = BottleneckBlock_3
    _bn_filters = [5, 8, 4, 4, 8, 8]
    _tr_blocks = [TransitionBlock_Down,
                  TransitionBlock_Plain,
                  TransitionBlock_Plain,
                  TransitionBlock_Plain,
                  TransitionBlock_Plain,
                  TransitionBlock_Up]
    _tr_filters = [4, 12, 12, 12, 4, 4]
    _us_pools = [2, 2, 2, 2, 2]

    db_blocks = DB_Blocks_Gen(_bn_block, _bn_filters,
                              _tr_blocks, _tr_filters, name)
    us_blocks = US_Blocks_Gen(_us_pools, name)
    return _Dense(x, db_blocks, us_blocks, name=name)


def Dense_rain(img_shape=(128, 128, 3), name='derain'):
    '''
    Multi-stream Dense Network
    '''
    if name is not None:
        name = name + '/'

    img_input = Input(img_shape, name=name + 'img_input' if name else None)
    label_input = Input(img_shape[:2] + (8,),
                        name=name + 'label_input' if name else None)

    residual = compose(
        Concatenate(name=name + 'concat1' if name else None),
        Conv2D(47, kernel_size=3, strides=1, padding='same',
               name=name + 'concat1/conv2d' if name else None),
        LeakyReLU(alpha=0.2, name=name + 'concat1/lrelu' if name else None),
        Concat_Samping_Block([32, 16, 8, 4], name=name +
                             'concat_sampling' if name else None),
        Conv2D(3, kernel_size=3, strides=1, padding='same',
               name=name + 'residual/conv2d' if name else None),
        Activation('tanh', name=name + 'residual/tanh' if name else None)
    )([Dense3(img_input), img_input, Dense2(img_input), Dense1(img_input), label_input])

    clear = compose(
        Subtract(name=name + 'clear/subtract' if name else None),
        Conv2D(8, kernel_size=7, strides=1, padding='same',
               name=name + 'clear/conv2d1' if name else None),
        LeakyReLU(alpha=0.2, name=name + 'clear/lrelu' if name else None),
        Conv2D(3, kernel_size=3, strides=1, padding='same',
               name=name + 'clear/conv2d2' if name else None),
        Activation('tanh', name=name + 'clear/tanh' if name else None)
    )([img_input, residual])

    model = Model(inputs=[img_input, label_input], outputs=[residual, clear])
    return model
