import tensorflow as tf
import numpy as np
from data_loading import fashion_mnist_loader as fashion_mnist 


def data_loading():
    
    x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names = fashion_mnist.data_loading()
    
    x_train, x_test = x_train.reshape((-1, 784)), x_test.reshape((-1, 784))
    
    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names