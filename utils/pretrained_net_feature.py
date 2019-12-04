from tensorflow.keras.models import Model

__all__ = ['pretrained_net_feature']

def pretrained_net_feature(pretrained_net, output_layers, input_shape=(128, 128, 3)):
    
    assert type(output_layers) in [int, list, str]
    if type(output_layers) in [int, str]:
        output_layers = [output_layers]
        
    pretrained_model = pretrained_net(include_top=False, 
                                      weights='imagenet', 
                                      input_shape=input_shape)
    layer_names = [layer.name for layer in pretrained_model.layers]
    output_list = []
    
    for output_layer in output_layers:
        if type(output_layer) is str:
            assert output_layer in layer_names
            output_layer = layer_names.index(output_layer)

        output_list.append(pretrained_model.layers[output_layer].output)
    
    model = Model(inputs=[pretrained_model.input], outputs=output_list)
    model.trainable = False
    return model
