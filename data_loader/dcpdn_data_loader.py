import cv2
import h5py
from glob import glob
import numpy as np
import os

class DCPDNDataLoader():
    def __init__(self, img_res=(512, 512), 
                 train_path='D:/GD/train512/', 
                 val_path='D:/GD/val512/'):
        self.dataset_name = self.__class__
        self.img_res = img_res
        
        self.train_path = train_path
        self.val_path = val_path
        self.data_type = '.h5'
        
    def load_data(self, batch_size=1, is_testing=False):
        if is_testing:
            file_names = glob(self.val_path + '*' + self.data_type)
        else:
            file_names = glob(self.train_path + '*' + self.data_type)
        
        batch_haze_base_names = np.random.choice(file_names, size=batch_size)

        ato_, gt_, haze_, trans_ = [], [], [], []
        
        for file_name in batch_haze_base_names:
            h5_result = self.load_h5(file_name)
            ato_.append(h5_result[0])
            gt_.append(h5_result[1])
            haze_.append(h5_result[2])
            trans_.append(h5_result[3])
            
        ato_ = np.array(ato_)
        gt_ = np.array(gt_)
        haze_ = np.array(haze_)
        trans_ = np.array(trans_)
        return ato_, gt_, haze_, trans_

    def load_batch(self, batch_size=1, is_testing=False):        
        if is_testing:
            file_names = glob(self.val_path + '*' + self.data_type)
        else:
            file_names = glob(self.train_path + '*' + self.data_type)

        self.n_batches = int(len(file_names) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        haze_file_names = np.random.choice(file_names, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_files = haze_file_names[i * batch_size : (i + 1) * batch_size]
            ato_, gt_, haze_, trans_ = [], [], [], []
            for file_name in batch_files:
                h5_result = self.load_h5(file_name)
                ato_.append(h5_result[0])
                gt_.append(h5_result[1])
                haze_.append(h5_result[2])
                trans_.append(h5_result[3])

            ato_ = np.array(ato_)
            gt_ = np.array(gt_)
            haze_ = np.array(haze_)
            trans_ = np.array(trans_)

            yield ato_, gt_, haze_, trans_

    def load_h5(self, path):
        file = h5py.File(path)
        ato, gt, haze, trans = np.array(file['ato']), np.array(file['gt']), np.array(file['haze']), np.array(file['trans'])
        file.close()
        return ato, gt, haze, trans
