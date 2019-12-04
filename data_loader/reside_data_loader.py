from .data_loader import DataLoader
import tensorflow as tf
import numpy as np


class RESIDEOTSDataLoader(DataLoader):
    def __init__(self,
                 img_shape=(256, 256),
                 clear_path='D:DataSet/RESIDE/OTS_ALPHA/clear_sl/',
                 haze_path='D:DataSet/RESIDE/OTS_ALPHA/haze/',
                 train_path='../data_loader/RESIDE_OTS_train.txt',
                 test_path='../data_loader/RESIDE_OTS_test.txt'):
        super(RESIDEOTSDataLoader, self).__init__(img_shape)
        self.dataset_name = self.__class__
        self.clear_path, self.haze_path = clear_path, haze_path
        self.train_path, self.test_path = train_path, test_path
        self.img_type = '.jpg'

    def train_data(self):
        basename_ds = tf.data.Dataset.from_tensor_slices(
            np.loadtxt(self.train_path, dtype=str))

        haze_train_img_ds = self._get_img_ds(basename_ds, self.haze_path)
        clear_train_img_ds = self._get_img_ds(basename_ds, self.clear_path)

        return haze_train_img_ds, clear_train_img_ds

    def test_data(self):
        basename_ds = tf.data.Dataset.from_tensor_slices(
            np.loadtxt(self.test_path, dtype=str))

        haze_train_img_ds = self._get_img_ds(basename_ds, self.haze_path)
        clear_train_img_ds = self._get_img_ds(basename_ds, self.clear_path)

        return haze_train_img_ds, clear_train_img_ds
