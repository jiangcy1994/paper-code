import cv2
from glob import glob
import numpy as np
import os

class RainyImageDataLoader():
    def __init__(self, img_res=(512, 512), 
                 clear_path='D:/DataSet/DetailsNet/{0}/ground_truth/', 
                 haze_path='D:/DataSet/DetailsNet/{0}/rainy_image/'):
        self.dataset_name = self.__class__
        self.img_res = img_res
        
        self.clear_path = clear_path
        self.haze_path = haze_path
        self.img_type = '.jpg'
        self.to_clear_base_name = lambda x: x.split('_')[0] + '.jpg'
        self.train_dir = 'training'
        self.test_dir = 'testing'
        
    def load_data(self, batch_size=1, is_testing=False):
        
        if is_testing:
            haze_path = self.haze_path.format(self.test_dir)
        else:
            haze_path = self.haze_path.format(self.train_dir)
        
        haze_base_names = [os.path.basename(path) for path in glob(haze_path + '*' + self.data_type)]
        batch_haze_base_names = np.random.choice(haze_base_names, size=batch_size)

        imgs = []
        for haze_base_name in batch_haze_base_names:
            img = self.load_img(haze_path + haze_base_name)
            if not is_testing and np.random.random() > 0.5:
                img = np.fliplr(img)
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing:
            clear_path = self.clear_path.format(self.test_dir)
            haze_path = self.haze_path.format(self.test_dir)
        else:
            clear_path = self.clear_path.format(self.train_dir)
            haze_path = self.haze_path.format(self.train_dir)
        
        haze_base_names = [os.path.basename(path) for path in glob(haze_path + '*' + self.img_type)]
        
        self.n_batches = int(len(haze_base_names) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        haze_base_names = np.random.choice(haze_base_names, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_haze = haze_base_names[i*batch_size : (i+1)*batch_size]
            clear_imgs, haze_imgs = [], []
            for haze_base_name in batch_haze:
                clear_img = self.load_img(clear_path + self.to_clear_base_name(haze_base_name))
                haze_img = self.load_img(haze_path + haze_base_name)

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
