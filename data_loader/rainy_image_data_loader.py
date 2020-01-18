from .data_loader import DataLoader
import tensorflow as tf
import numpy as np
import os

class RainyImageDataLoader(DataLoader):
    def __init__(self, img_shape=(256, 256), 
                 clear_path='D:/DataSet/DetailsNet/{0}/ground_truth/', 
                 rain_path='D:/DataSet/DetailsNet/{0}/rainy_image/'):
        super(RainyImageDataLoader, self).__init__(img_shape)
        self.dataset_name = self.__class__
        
        self.clear_path = clear_path
        self.rain_path = rain_path
        self.img_type = '.jpg'
        self.to_clear_base_name = lambda x: x.split('_')[0] + self.img_type

    def train_data(self):
        format_str = 'training'
        rain_basename_list = os.listdir(self.rain_path.format(format_str))
        rain_basename_ds = tf.data.Dataset.from_tensor_slices(rain_basename_list)
        
        clear_basename_list = [self.to_clear_base_name(rain_basename) for rain_basename in rain_basename_list]
        clear_basename_ds = tf.data.Dataset.from_tensor_slices(clear_basename_list)

        rain_train_img_ds = self._get_img_ds(rain_basename_ds, self.rain_path.format(format_str))
        clear_train_img_ds = self._get_img_ds(clear_basename_ds, self.clear_path.format(format_str))

        return rain_train_img_ds, clear_train_img_ds
  
    def test_data(self):
        format_str = 'testing'
        rain_basename_list = os.listdir(self.rain_path.format(format_str))
        rain_basename_ds = tf.data.Dataset.from_tensor_slices(rain_basename_list)
        
        clear_basename_list = [self.to_clear_base_name(rain_basename) for rain_basename in rain_basename_list]
        clear_basename_ds = tf.data.Dataset.from_tensor_slices(clear_basename_list)

        rain_train_img_ds = self._get_img_ds(rain_basename_ds, self.rain_path.format(format_str))
        clear_train_img_ds = self._get_img_ds(clear_basename_ds, self.clear_path.format(format_str))
        
        rain_train_img_ds = self._get_img_ds(rain_basename_ds, self.rain_path.format(format_str))
        clear_train_img_ds = self._get_img_ds(clear_basename_ds, self.clear_path.format(format_str))

        return rain_train_img_ds, clear_train_img_ds
