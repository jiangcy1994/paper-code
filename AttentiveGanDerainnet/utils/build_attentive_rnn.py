from compose import *
from keras.layers import Concatenate, Conv2D, Input
from keras.models import Model
from .residual_block import *
from .conv_lstm import *

__all__ = ['build_attentive_rnn']

def build_attentive_rnn(img_shape):
    
    img_input = Input(img_shape)
    init_attention_map = Input(img_shape[:-1] + (1,))
    init_cell_state = Input(img_shape[:-1] + (32,))
    
    attention_map = init_attention_map
    cell_state = init_cell_state
    
    attention_map_list = []

    for i in range(4):
        attention_input = Concatenate(name='concat_{0}'.format(i + 1))([img_input, attention_map])
        
        conv_feats = residual_block(input_tensor=attention_input,
                                    name='residual_block_{0}'.format(i + 1))
        lstm_ret = conv_lstm(input_tensor=conv_feats,
                             input_cell_state=cell_state,
                             name='conv_lstm_block_{0}'.format(i + 1))
        attention_map = lstm_ret['attention_map']
        cell_state = lstm_ret['cell_state']
        lstm_feats = lstm_ret['lstm_feats']

        attention_map_list.append(lstm_ret['attention_map'])
        
#     attention_maps = Concatenate(name='attention_maps')(attention_map_list)

    return Model(inputs=[img_input, init_attention_map, init_cell_state],
                 outputs=[attention_map, lstm_feats, 
                          attention_map_list[0], attention_map_list[1], attention_map_list[2], attention_map_list[3]], 
                 name='AttentionRNN')
