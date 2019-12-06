from .data_loader import DataLoader
import h5py
import tensorflow as tf
import numpy as np


class RESIDEOTSDataLoader(DataLoader):
    def __init__(self,
                 img_shape=(256, 256),
                 clear_path='D:DataSet/RESIDE/OTS_ALPHA/clear_sl/',
                 depth_path='D:DataSet/RESIDE/OTS_ALPHA/depth/'
                 haze_path='D:DataSet/RESIDE/OTS_ALPHA/haze/',
                 train_path='../data_loader/RESIDE_OTS_train.txt',
                 test_path='../data_loader/RESIDE_OTS_test.txt'):
        super(RESIDEOTSDataLoader, self).__init__(img_shape)
        self.dataset_name = self.__class__
        self.clear_path, self.depth_path, self.haze_path = clear_path, depth_path, haze_path
        self.train_path, self.test_path = train_path, test_path
        self.img_type = '.jpg'

    def _get_atmos_ds():
        atmos_ds = ds.map(map_func=lambda x: x.split('_')[1])
        return trans_ds

    def _get_trans_ds():
        beta_ds = ds.map(map_func=lambda x: x.split('_')[2])
        depth_path_ds = ds.map(
            map_func=lambda x: self.depth_path + x.split('_')[0] + '.mat')
        depth_ds = ds.map(map_func=lambda x: h5py.File(
            filename, 'r')['depth'][:].transpose())
        trans_ds = tf.data.Dataset.zip((beta_ds, depth_ds))
        trans_ds = trans_ds.map(map_func=lambda beta,
                                depth: tf.exp(-1 * beta * depth))
        return trans_ds

    def train_data(self, include_trans_atmos=False):
        basename_ds = tf.data.Dataset.from_tensor_slices(
            np.loadtxt(self.train_path, dtype=str))

        if include_trans_atmos:
            pass
        else:
            haze_train_img_ds = self._get_img_ds(basename_ds, self.haze_path)
            clear_train_img_ds = self._get_img_ds(basename_ds, self.clear_path)

            return haze_train_img_ds, clear_train_img_ds

    def test_data(self):
        basename_ds = tf.data.Dataset.from_tensor_slices(
            np.loadtxt(self.test_path, dtype=str))

        haze_train_img_ds = self._get_img_ds(basename_ds, self.haze_path)
        clear_train_img_ds = self._get_img_ds(basename_ds, self.clear_path)

        return haze_train_img_ds, clear_train_img_ds
