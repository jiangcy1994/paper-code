from compose import *
from tensorflow.keras.layers import Activation, Add, Conv2D, Input, LeakyReLU, Multiply

__all__ = ['conv_lstm']

conv_activat = lambda activat, block_name, name, out_channel=32: compose(
    Conv2D(out_channel, kernel_size=3, strides=1, padding='same', use_bias=False, name=name + 'conv_' + block_name),
    Activation(activat, name=name + 'activat_' + block_name)
)

def conv_lstm(input_tensor, input_cell_state, name):
    
    if type(name) is str:
        name = name + '/'
    
    sigmoid_input = conv_activat('sigmoid', 'i', name)(input_tensor)
    sigmoid_forget = conv_activat('sigmoid', 'f', name)(input_tensor)
    tanh_cell_state = conv_activat('tanh', 'c', name)(input_tensor)
    sigmoid_output = conv_activat('sigmoid', 'o', name)(input_tensor)
    
    cell_state = Add(name=name + 'add_c')([
        Multiply(name=name + 'mul_f_c')([sigmoid_forget, input_cell_state]),
        Multiply(name=name + 'mul_i_c')([sigmoid_input, tanh_cell_state])
    ])
        
    lstm_feats = Multiply(name=name + 'mul_lf')([
        sigmoid_output,
        Activation('tanh', name=name + 'tanh_c')(cell_state)
    ])
    
    attention_map = conv_activat('sigmoid', 'attention_map', name, out_channel=1)(lstm_feats)
    
    ret = {
        'attention_map': attention_map,
        'cell_state': cell_state,
        'lstm_feats': lstm_feats
    }
    return ret