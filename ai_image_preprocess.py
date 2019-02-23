import cv2
import numpy as np

class ImagePreprocessor:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def downsize(self, img_arry):
        return cv2.resize(img_arry, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def rgb2gray(self, img_arr):
        return np.dot(img_arr[...,:3], [0.299, 0.587, 0.114])

    def normalize(self, img_arr):
        return np.divide(img_arr, 255.0)

    def preprocess(self, img_arr):
        downsized = self.downsize(img_arr)
        gray = self.rgb2gray(downsized)
        normalized = self.normalize(gray)
        return normalized
        #extra_dim = normalized[..., np.newaxis]
        #for i in range(2):
        #    extra_dim = np.append(extra_dim, extra_dim, axis=2)
        #return np.stack(extra_dim)
    