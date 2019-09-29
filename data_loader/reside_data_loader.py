import cv2
from glob import glob
import numpy as np
import os

class RESIDEOTSDataLoader():
    def __init__(self, img_res=(256, 256)):
        self.dataset_name = self.__class__
        self.img_res = img_res
        
        self.clear_path = 'D:/DataSet/RESIDE/OTS_ALPHA/clear/clear_images/'
        self.haze_path = 'D:/DataSet/RESIDE/OTS_ALPHA/haze/OTS/'
        self.img_type = '.jpg'
        self.to_clear_base_name = lambda x: x.split('_')[0] + '.jpg'
        self.train_range = range(1, 5400)
        self.valid_range = range(5401, 7200)
        self.test_range = range(7201, 10000)
        
    def load_data(self, batch_size=1, is_testing=False):
        haze_base_names = [os.path.basename(path) for path in glob(self.haze_path + '*' + self.img_type)]
        if is_testing:
            haze_base_names = [*filter(lambda x: int(x.split('_')[0]) in self.test_range, haze_base_names)]
        else:
            haze_base_names = [*filter(lambda x: int(x.split('_')[0]) not in self.test_range, haze_base_names)]     
        
        batch_haze_base_names = np.random.choice(haze_base_names, size=batch_size)

        imgs = []
        for haze_base_name in batch_haze_base_names:
            img = self.load_img(self.haze_path + haze_base_name)
            if not is_testing and np.random.random() > 0.5:
                img = np.fliplr(img)
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        haze_base_names = [os.path.basename(path) for path in glob(self.haze_path + '*' + self.img_type)]
        
        if is_testing:
            haze_base_names = [*filter(lambda x: int(x.split('_')[0]) in self.valid_range, haze_base_names)]
        else:
            haze_base_names = [*filter(lambda x: int(x.split('_')[0]) in self.train_range, haze_base_names)]     

        self.n_batches = int(len(haze_base_names) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        haze_base_names = np.random.choice(haze_base_names, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_haze = haze_base_names[i*batch_size : (i+1)*batch_size]
            clear_imgs, haze_imgs = [], []
            for haze_base_name in batch_haze:
                clear_img = self.load_img(self.clear_path + self.to_clear_base_name(haze_base_name))
                haze_img = self.load_img(self.haze_path + haze_base_name)

                if not is_testing and np.random.random() > 0.5:
                    clear_img = np.fliplr(clear_img)
                    haze_img = np.fliplr(haze_img)

                clear_imgs.append(clear_img)
                haze_imgs.append(haze_img)

            clear_imgs = np.array(clear_imgs)
            haze_imgs = np.array(haze_imgs)

            yield clear_imgs, haze_imgs

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res)
        img = img / 127.5 - 1.
        return img[:, :, :]

    def imread(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)
