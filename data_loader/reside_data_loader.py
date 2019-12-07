from .data_loader import DataLoader
import h5py
import tensorflow as tf
import numpy as np


class RESIDEOTSDataLoader(DataLoader):
    def __init__(self,
                 img_shape=(256, 256),
                 clear_path='D:DataSet/RESIDE/OTS_ALPHA/clear_sl/',
                 depth_path='D:DataSet/RESIDE/OTS_ALPHA/depth/',
                 haze_path='D:DataSet/RESIDE/OTS_ALPHA/haze/',
                 train_path='../data_loader/RESIDE_OTS_train.txt',
                 test_path='../data_loader/RESIDE_OTS_test.txt'):
        super(RESIDEOTSDataLoader, self).__init__(img_shape)
        self.dataset_name = self.__class__
        self.clear_path, self.depth_path, self.haze_path = clear_path, depth_path, haze_path
        self.train_path, self.test_path = train_path, test_path
        self.img_type = '.jpg'
        h5py.get_config().default_file_mode = 'r'

    def _gen_gen(self, data_list, gen_func):
        def gen():
            for data in data_list:
                yield gen_func(data)
        return gen

    def _get_atmos_ds(self, base_name_list):

        def _gen_atmos(base_name):
            return tf.fill([self.img_shape[0], self.img_shape[1], 1], float(base_name.split('_')[1]))

        return tf.data.Dataset.from_generator(
            self._gen_gen(base_name_list, _gen_atmos),
            tf.float32,
            tf.TensorShape([self.img_shape[0], self.img_shape[1], 1])
        )

    def _get_trans_ds(self, base_name_list):

        def _get_trans(base_name):
            beta = float(base_name.split('_')[2][:-4])
            depth = tf.keras.utils.HDF5Matrix(
                self.depth_path + base_name.split('_')[0] + '.mat', 'depth')
            depth = tf.image.resize(
                tf.expand_dims(depth, axis=-1),
                self.img_shape
            )
            trans = tf.exp(-1 * beta * depth)
            trans = tf.concat([trans, trans, trans], axis=-1)
            return trans

        return tf.data.Dataset.from_generator(
            self._gen_gen(base_name_list, _get_trans),
            tf.float32,
            tf.TensorShape([self.img_shape[0], self.img_shape[1], 3])
        )

    def train_data(self, include_trans_atmos=False):
        basename_list = np.loadtxt(self.train_path, dtype=str)
        basename_ds = tf.data.Dataset.from_tensor_slices(basename_list)

        if include_trans_atmos:
            haze_train_img_ds = self._get_img_ds(basename_ds, self.haze_path)
            clear_train_img_ds = self._get_img_ds(basename_ds, self.clear_path)
            trans_ds = self._get_trans_ds(basename_list)
            atmos_ds = self._get_atmos_ds(basename_list)

            return haze_train_img_ds, clear_train_img_ds, trans_ds, atmos_ds

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
