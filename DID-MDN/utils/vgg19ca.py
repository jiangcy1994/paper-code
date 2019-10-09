from keras.applications import vgg19
from keras.layers import AvgPool2D, BatchNormalization, Conv2D, Dense, Flatten, Input, ReLU, UpSampling2D
from keras.models import Model

__all__ = ['VGG19ca']

def VGG19ca(img_shape=(128,128,3)):
    vgg19_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=img_shape)

    # Build Model
    inp = vgg19_model.input
    
    # Blocks in VGG19
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inp)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(24, kernel_size=3, strides=1, padding='same')(x)
    out = ReLU()(x)
    out = AvgPool2D(7)(out)
    
    #Dense
    out = Flatten()(out)
    out = Dense(512)(out)
    out = ReLU()(out)
    out = Dense(4)(out)
    model = Model(inputs=[inp], outputs=[out])
    
    # Transfer Weight
    model.layers[1].set_weights(vgg19_model.layers[1].get_weights())
    return model
