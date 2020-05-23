from scipy.io import loadmat
import urllib
import tensorflow as tf
import numpy as np

def get_faces_dataset():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/larajuse/share/master/mlds6/S10/faces.mat", "faces.mat")
    data = np.swapaxes(loadmat("faces.mat")["faces"].T.reshape(-1, 64, 64, 1), 1, 2)/255.0
    data = np.concatenate([data, data[:16]], axis=0)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, shear_range=5,
                                                               height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
    def ae_gen(gen):
        while True:
            data = next(gen)
            yield (data, data)
    return ae_gen(data_gen.flow(data))