import tensorflow as tf
import numpy as np
import os
from data_loading.HvassLabs import cifar10  


def data_loading():
    
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/CIFAR-10"):
        os.mkdir("data/CIFAR-10")

    # download and extract if not done yet
    # data is downloaded from data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    #                    to data_path  = "data/CIFAR-10/"
    cifar10.data_path = "data/CIFAR-10/"
    cifar10.maybe_download_and_extract()

    # load data
    x_train, y_train_cls, y_train = cifar10.load_training_data()
    x_test, y_test_cls, y_test = cifar10.load_test_data()
    cls_names = cifar10.load_class_names()

    x_train = x_train.astype(np.float32)
    y_train_cls = y_train_cls.astype(np.int32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test_cls = y_test_cls.astype(np.int32)
    y_test = y_test.astype(np.float32)

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names