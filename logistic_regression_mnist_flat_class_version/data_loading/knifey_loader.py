import numpy as np
import os
from data_loading.HvassLabs import knifey


def data_loading():

    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/knifey-spoony"):
        os.mkdir("data/knifey-spoony")

    # download and extract if not done yet
    knifey.data_path = "data/knifey-spoony/"
    knifey.maybe_download_and_extract()

    dataset = knifey.load()
    x_train, y_train_cls, y_train = dataset.get_training_set()
    x_test, y_test_cls, y_test = dataset.get_test_set()
    cls_names = dataset.class_names

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_train_cls = y_train_cls.astype(np.int32)
    y_test_cls = y_test_cls.astype(np.int32)

    return x_train, y_train, y_train_cls, x_test, y_test, y_test_cls, cls_names
