import tensorflow as tf
import numpy as np


def data_loading():

    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    x_grid, y_grid = np.meshgrid(x, y)
    z0_grid = x_grid**2 + y_grid**2

    np.random.seed(1)
    
    ep_train = np.random.randn(x_grid.shape[0], x_grid.shape[1])
    ep_test = np.random.randn(x_grid.shape[0], x_grid.shape[1])
    
    z_grid_train = z0_grid + 0.3 * ep_train
    z_grid_test = z0_grid + 0.3 * ep_test

    X = x_grid.reshape([-1, 1])
    Y = y_grid.reshape([-1, 1])
    Z_train = z_grid_train.reshape([-1, 1])
    Z_test = z_grid_test.reshape([-1, 1])

    x_train = np.hstack([X, Y, Z_train]).astype(np.float32)
    x_test = np.hstack([X, Y, Z_test]).astype(np.float32)

    return x_train, x_test

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls