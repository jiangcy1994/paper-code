from tensorflow.keras.applications import vgg16
from pretrained_net_feature import *

__all__ = ['vgg16_feature_net']

vgg16_feature_net = lambda img_shape=(256, 256, 3): pretrained_net_feature(
    pretrained_net=vgg16.VGG16,
    output_layers=['block1_conv2', 'block2_conv2'],
    input_shape=img_shape
)
