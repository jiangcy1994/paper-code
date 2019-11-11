import cv2
from glob import glob
import itertools
import numpy as np
import os

labels = ['Rain_Light', 'Rain_Medium', 'Rain_Heavy']

class DIDMDNDataLoader():
    def __init__(self, img_res=(512, 512), 
                 img_path='D:/DataSet/RESIDE/{0}/training/{1}/'):
        self.dataset_name = self.__class__
        self.img_res = img_res
        
        self.img_path = img_path
        self.img_type = '.jpg'
        self.train_dir = 'training'
        self.test_dir = 'testing'
        
    def load_data(self, batch_size=1, is_testing=False):
        if is_testing:
            img_path = self.img_path.format(self.test_dir, '')
            rain_base_names = ['{0}.'.format(i) + self.img_type for i in range(4000) ]
        else:
            img_path = self.img_path.format(self.train_dir, '{0}')
            rain_base_names = ['{0}.'.format(i) + self.img_type for i in range(1200) ]
        
        batch_rain_base_names = np.random.choice(rain_base_names, size=batch_size)

        clear_imgs, rain_imgs, labels = [], [], []
        
        if is_testing:
            for rain_base_name in batch_haze:
                clear_img, rain_img = self.load_img(img_path + rain_base_name)
                if not is_testing and np.random.random() > 0.5:
                    clear_img = np.fliplr(clear_img)
                    rain_img = np.fliplr(rain_img)

                clear_imgs.append(clear_img)
                rain_imgs.append(rain_img)
                labels.append(0)
        else:
            for rain_base_name, (label_id, label_name) in itertools.product(batch_haze, enumerate(labels)):
                clear_img, rain_img = self.load_img(img_path.format(label_name) + rain_base_name)

                if not is_testing and np.random.random() > 0.5:
                    clear_img = np.fliplr(clear_img)
                    rain_img = np.fliplr(rain_img)

                clear_imgs.append(clear_img)
                rain_imgs.append(rain_img)
                labels.append(label_id)

        clear_imgs = np.array(clear_imgs)
        rain_imgs = np.array(rain_imgs)
        labels = np.array(labels)

        return rain_imgs, clear_imgs, labels

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing:
            img_path = self.img_path.format(self.test_dir, '')
            rain_base_names = ['{0}.'.format(i) + self.img_type for i in range(4000) ]
        else:
            img_path = self.img_path.format(self.train_dir, '{0}')
            rain_base_names = ['{0}.'.format(i) + self.img_type for i in range(1200) ]
        
        self.n_batches = int(len(rain_base_names) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        rain_base_names = np.random.choice(rain_base_names, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_haze = rain_base_names[i*batch_size : (i+1)*batch_size]
            clear_imgs, rain_imgs, labels = [], [], []
            
            if is_testing:
                for rain_base_name in batch_haze:
                    clear_img, rain_img = self.load_img(img_path + rain_base_name)

                    if not is_testing and np.random.random() > 0.5:
                        clear_img = np.fliplr(clear_img)
                        rain_img = np.fliplr(rain_img)

                    clear_imgs.append(clear_img)
                    rain_imgs.append(rain_img)
                    labels.append(0)
            else:
                for rain_base_name, (label_id, label_name) in itertools.product(batch_haze, enumerate(labels)):
                    clear_img, rain_img = self.load_img(img_path.format(label_name) + rain_base_name)

                    if not is_testing and np.random.random() > 0.5:
                        clear_img = np.fliplr(clear_img)
                        rain_img = np.fliplr(rain_img)

                    clear_imgs.append(clear_img)
                    rain_imgs.append(rain_img)
                    labels.append(label_id)

            clear_imgs = np.array(clear_imgs)
            rain_imgs = np.array(rain_imgs)
            labels = np.array(labels)

            yield rain_imgs, clear_imgs, labels

    def load_img(self, path):
        img = self.imread(path)
        
        half_wid = img.shape[1] // 2
        
        img_rain = img[:,:half_wid]
        img_clear = img[:,half_wid:]
        
        img_rain = cv2.resize(img_rain, self.img_res)
        img_clear = cv2.resize(img_clear, self.img_res)
        
        img_rain = img_rain / 127.5 - 1.
        img_clear = img_clear / 127.5 - 1.
        return img_rain, img_clear

    def imread(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)
