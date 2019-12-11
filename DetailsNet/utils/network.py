import compose
import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Input, ReLU

__all__ = ['inference']


def inference(images_shape, num_feature=16, kernel_size=3):
    num_channels = images_shape[2]

    def conv2d(filters, layer_id): return Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(1e-10), name='layer_{0}/conv'.format(layer_id))

    def bn(layer_id): return BatchNormalization(
        name='layer_{0}/bn'.format(layer_id))

    def relu(layer_id): return ReLU(name='layer_{0}/relu'.format(layer_id))

    def base_layer(filters, layer_id): return compose(
        conv2d(filters, layer_id),
        bn(layer_id),
        relu(layer_id)
    )

    images = Input(images_shape, name='images_input')
    detail = Input(images_shape, name='detail_input')


#     base = guided_filter(inp, inp, 15, 1, nhwc=True) # using guided filter for obtaining base layer
#     detail = images - base   # detail layer
    output_shortcut = base_layer(num_feature, 1)

    #  layers 2 to 25
    for i in range(1, 13):

        output = compose(
            base_layer(num_feature, i * 2),
            base_layer(num_feature, i * 2 + 1)
        )(output_shortcut)

        output_shortcut = Add(name='add_{0}'.format(i))(
            [output_shortcut, output])

    # layer 26
    neg_residual = compose(
        conv2d(num_channels, 26),
        bn(26)
    )(output_shortcut)

    final_out = Add(name='add_final')([images, neg_residual])

    return tf.keras.models.Model(inputs=[images, detail], outputs=[final_out])
