from compose import *
from tensorflow.keras.layers import Conv2D, Input, ReLU
from tensorflow.keras.models import Model

def inference(input_shape=(33, 33, 1)):
    layers = compose(
        Conv2D(64, kernel_size=9, strides=1, padding='valid', use_bias=True),
        ReLU(),
        Conv2D(32, kernel_size=1, strides=1, padding='valid', use_bias=True),
        ReLU(),
        Conv2D(3, kernel_size=5, strides=1, padding='valid', use_bias=True),
    )

    inputs = Input(input_shape)

    return Model(inputs=[inputs], outputs=[layers(inputs)])
