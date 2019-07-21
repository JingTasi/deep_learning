import tensorflow as tf
import numpy as np


def data_loading():
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape((-1, 28, 28, 1)), x_test.reshape((-1, 28, 28, 1))
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    y_train_cls = y_train.copy().astype(np.int32)
    y_test_cls = y_test.copy().astype(np.int32)

    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    cls_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names
