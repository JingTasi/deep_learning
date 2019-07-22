import data_loading.cifar10_loader as mnist


def data_loading():

    x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names = mnist.data_loading()

    x_train, x_test = x_train.reshape((-1, 3072)), x_test.reshape((-1, 3072))

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names
