import data_loading.mnist_loader as mnist


def data_loading():

    x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names = mnist.data_loading()

    x_train, x_test = x_train.reshape((-1, 784)), x_test.reshape((-1, 784))

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names
