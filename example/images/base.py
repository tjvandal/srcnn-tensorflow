import os, sys
import scipy.misc
import numpy as np
import tensorflow as tf
import requests, zipfile
import StringIO
from PIL import Image

def _download_images(dir):
    url = 'https://github.com/jbhuang0604/SelfExSR/archive/master.zip'
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall()
    for name in z.namelist():
        if '.png' in name:
            to_file = os.path.join(dir, name)
            if not os.path.exists(os.path.dirname(to_file)):
                os.makedirs(os.path.dirname(to_file))
            img = Image.open(StringIO.StringIO(z.read(name)))
            img.save(os.path.join(dir, name))

class SuperResData:
    def __init__(self, upscale_factor=2, imageset='Set5'):
        self.upscale_factor = upscale_factor
        self.imageset = imageset
        self._base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(self._base_dir):
            _download_images(self._base_dir)
        self.data_dir = os.path.join(self._base_dir, 'SelfExSR-master/data/', imageset,
                                     'image_SRF_%i' % self.upscale_factor)

    def read(self):
        hr_images = {}
        lr_images = {}
        for i, f in enumerate(os.listdir(self.data_dir)):
            img = scipy.misc.imread(os.path.join(self.data_dir, f)) / 255.
            if "HR" in f:
                hr_images["".join(f.split("_")[:3])] = img
            elif "LR" in f:
                lr_images["".join(f.split("_")[:3])] = img
        lr_keys = sorted(lr_images.keys())
        hr_keys = sorted(hr_images.keys())
        assert lr_keys == hr_keys
        for k in hr_keys:
            yield lr_images[k], hr_images[k]

    def make_patches(self, patch_size=15, stride=8):
        """
        Args:
            patch_size: size of low-resolution subimages
            stride: step length between subimages
        """
        X_sub = []
        Y_sub = []
        for x, y in self.read():
            if len(x.shape) != 3:
                continue
            h, w, _ = x.shape
            for i in np.arange(0, h, stride):
                for j in np.arange(0, w, stride):
                    hi_low, hi_high = i, i+patch_size
                    wi_low, wi_high = j, j+patch_size
                    if (hi_high > h) or (wi_high > w):
                        continue
                    X_sub.append(x[np.newaxis,hi_low:hi_high,wi_low:wi_high])
                    Y_sub.append(y[np.newaxis,self.upscale_factor*hi_low:self.upscale_factor*hi_high,
                                   self.upscale_factor*wi_low:self.upscale_factor*wi_high])
        X_sub = np.concatenate(X_sub, axis=0)
        Y_sub = np.concatenate(Y_sub, axis=0)
        return X_sub, Y_sub

    def tf_shuffle_pipeline(self, X, Y, batch_size):
        data = [tf.constant(X, name='x', dtype=tf.float32), tf.constant(Y, name='y', dtype=tf.float32)]
        batch = tf.train.shuffle_batch(data,
                                      batch_size=batch_size,
                                      num_threads=8,
                                      capacity=1000,
                                      min_after_dequeue=500,
                                      enqueue_many=True)
        return batch[0], batch[1]

    def tf_patches(self, batch_size=20, patch_size=15,
                            stride=8, scope=None):
        X, Y = self.make_patches(patch_size=patch_size, stride=stride)
        with tf.variable_scope(scope, "shuffle_patches"):
            return self.tf_shuffle_pipeline(X, Y, batch_size=batch_size)

    def get_images(self):
        """
        This method returns two lists of tensorflow constants of low-res and high-res
                images (input and label)
        """
        X, Y = [], []
        for x, y in self.read():
            if len(x.shape) != 3:
                continue
            X.append(x[np.newaxis].astype(np.float32))
            Y.append(y[np.newaxis].astype(np.float32))
        return X, Y

if __name__ == '__main__':
    d = SuperResData(imageset='Set5')
    print d.tf_images()
