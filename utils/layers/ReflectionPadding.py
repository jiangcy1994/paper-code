import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils

class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
      
    This layer can add rows and columns of reflction
    at the top, bottom, left and right side of an image tensor.

    Arguments:
      padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        - If int: the same symmetric padding
          is applied to height and width.
        - If tuple of 2 ints:
          interpreted as two different
          symmetric padding values for height and width:
          `(symmetric_height_pad, symmetric_width_pad)`.
        - If tuple of 2 tuples of 2 ints:
          interpreted as
          `((top_pad, bottom_pad), (left_pad, right_pad))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, rows, cols)`
  
    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, padded_rows, padded_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, padded_rows, padded_cols)`
    """
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])


    def call(self, inputs):
        
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], list(self.padding[0]), list(self.padding[1])]
        else:
            pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]
        return tf.pad(inputs, pattern, mode="REFLECT")
        
    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
