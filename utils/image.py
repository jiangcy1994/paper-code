import cv2
import numpy as np

def load_img(self, path, img_shape, interpolation=cv2.INTER_LINEAR):
    img = self.imread(path)
    img = cv2.imresize(img, img_shape, interpolation=interpolation)
    img = img / 127.5 - 1.
    return img[np.newaxis, :, :, :]

def imread(self, path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)
