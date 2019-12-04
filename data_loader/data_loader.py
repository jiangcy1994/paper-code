import numpy as np
import tensorflow as tf
import sys
sys.path.append('../utils/')

import image

class DataLoader():
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def train_data(self):
        pass

    def test_data(self):
        pass

    def _get_img_ds(self, ds, prefix):
        img_file_ds = ds.map(map_func=lambda x: prefix + x)
        img_ds = img_file_ds.map(
            map_func=lambda path: image.load_and_preprocess_image(path, (256, 256)))
        return img_ds
