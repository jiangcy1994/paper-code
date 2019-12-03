import tensorflow as tf

def preprocess_image(image, img_shape):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0

    return image

def load_and_preprocess_image(path, img_shape):
    image = tf.io.read_file(path)
    return preprocess_image(image, img_shape)
