from keras.applications import vgg16
from keras.models import Model

__all__ = ['VGG16_Feature']

def VGG16_Feature(img_shape=(256, 256, 3)):
    vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=img_shape)
    model = Model(inputs=[vgg16_model.input], outputs=[
        vgg16_model.layers[1].output,
        vgg16_model.layers[2].output,
        vgg16_model.layers[4].output,
        vgg16_model.layers[5].output,
        vgg16_model.layers[7].output,
        vgg16_model.layers[8].output,
        vgg16_model.layers[9].output
    ])
    model.trainable = False
    return model
