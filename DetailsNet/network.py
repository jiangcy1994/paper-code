from keras.initializers import glorot_uniform
from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Input, ReLU
from keras.models import Model
from keras.regularizers import l2

def inference(images_shape, is_training, num_feature=16, KernelSize=3):
    initializer = glorot_uniform()
    regularizer = l2(1e-10)
    num_channels = images_shape[2]
    
    conv2d = lambda layer_id: Conv2D(filters=num_feature, kernel_size=KernelSize, padding='same', 
                                     kernel_initializer=initializer, kernel_regularizer=regularizer,
                                     name='layer_%d/conv_%d' % (layer_id, layer_id))
    bn = lambda layer_id: BatchNormalization(name='layer_%d/bn_%d' % (layer_id, layer_id))
    relu = lambda layer_id: ReLU(name='layer_%d/relu_%d' % (layer_id, layer_id))
    
    images = Input(images_shape, name='images_input')
    detail = Input(images_shape, name='detail_input')
    
    
#     base = guided_filter(inp, inp, 15, 1, nhwc=True) # using guided filter for obtaining base layer
#     detail = images - base   # detail layer
    
    output = conv2d(1)(detail)
    output = bn(1)(output, training=is_training)
    output_shortcut = relu(1)(output)
    
    #  layers 2 to 25
    for i in range(12):
        layer_id = i * 2 + 2
        output = conv2d(layer_id)(output_shortcut)
        output = bn(layer_id)(output, training=is_training)
        output = relu(layer_id)(output)

        layer_id = i * 2 + 3
        output = conv2d(layer_id)(output)
        output = bn(layer_id)(output, training=is_training)
        output = relu(layer_id)(output)

        output_shortcut = Add(name='add_%d'%i)([output_shortcut, output])

    # layer 26
    output = Conv2D(filters=num_channels, kernel_size=KernelSize, padding='same', 
                    kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name='layer_26/conv_26')(output_shortcut)
    neg_residual = bn(26)(output, training=is_training)
    
    final_out = Add(name='add_final')([images, neg_residual])
    
    return Model(inputs=[images, detail], outputs=[final_out])
    