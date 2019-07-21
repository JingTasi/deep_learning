import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread, cm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image, display
from copy import deepcopy
from sklearn import datasets

# import inception                             # module from Hvass Labs
# import cache                                 # module from Hvass Labs
# import dataset                               # module from Hvass Labs
# import download                              # module from Hvass Labs
# import cifar10                               # module from Hvass Labs
# import knifey                                # module from Hvass Labs
# from inception import NameLookup             # module from Hvass Labs
# from inception import transfer_values_cache  # module from Hvass Labs


def load_two_asset_daily_return_data():

    start = "2017-01-01"
    end = "2017-12-31"
    dates = pd.date_range(start, end)

    ticker_list = ["SPY", "WMT"]
    benchmark = "SPY"

    data_dir = "./data/dow30"

    for data_type in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:

        df = pd.DataFrame(index=dates)
        for ticker in ticker_list:
            csv_file_path = os.path.join(data_dir, ticker + ".CSV")

            df_temp = pd.read_csv(csv_file_path,
                                  index_col="Date",
                                  parse_dates=True,
                                  usecols=["Date", data_type],
                                  na_values=["null"])

            df_temp = df_temp.rename(columns={data_type: ticker})
            df = df.join(df_temp)

            if ticker == benchmark:
                df = df.dropna(subset=[benchmark])

        if data_type == "Open":
            df_open = df
        if data_type == "High":
            df_high = df
        if data_type == "Low":
            df_low = df
        if data_type == "Close":
            df_close = df
        if data_type == "Adj Close":
            df_adj_close = df
        if data_type == "Volume":
            df_volume = df

    df_daily_return = df_adj_close.pct_change()

    spy = np.array(df_daily_return.SPY).reshape(-1, 1).astype(np.float32)[1:]
    wmt = np.array(df_daily_return.WMT).reshape(-1, 1).astype(np.float32)[1:]

    data = (spy, wmt)

    return data


if __name__ == '__main__':
    print("load_two_asset_daily_return_data")
    print()

    spy, wmt = load_two_asset_daily_return_data()
    print(type(spy))
    print(type(wmt))
    print(spy.shape)
    print(wmt.shape)

    plt.plot(spy, wmt, 'o')
    plt.xlabel('SPY daily return')
    plt.ylabel('WMT daily return')
    plt.show()


def load_2_factor_diabetes_data():
    diabetes = datasets.load_diabetes()

    x_train = diabetes.data[:, 2:4][:-20, :].astype(np.float32)  # use two features
    x_test = diabetes.data[:, 2:4][-20:, :].astype(np.float32)  # use two features
    y_train = diabetes.target[:-20].reshape(-1, 1).astype(np.float32)
    y_test = diabetes.target[-20:].reshape(-1, 1).astype(np.float32)

    data = (x_train, y_train, x_test, y_test)

    return data


if __name__ == '__main__':
    print("load_2_factor_diabetes_data")
    print()

    data = load_2_factor_diabetes_data()

    x_train, x_test, y_train, y_test = data

    print("x_train.shape     : ", x_train.shape)
    print("x_test.shape      : ", x_test.shape)
    print("y_train.shape     : ", y_train.shape)
    print("y_test.shape      : ", y_test.shape)
    print()

    print("x_train.dtype     : ", x_train.dtype)
    print("x_test.dtype      : ", x_test.dtype)
    print("y_train.dtype     : ", y_train.dtype)
    print("y_test.dtypee     : ", y_test.dtype)
    print()

    print("type(x_train)     : ", type(x_train))
    print("type(x_test)      : ", type(x_test))
    print("type(y_train)     : ", type(y_train))
    print("type(y_test)      : ", type(y_test))
    print()

    print("min and max of x_train     : ", np.min(x_train), np.max(x_train))
    print("min and max of x_test      : ", np.min(x_test), np.max(x_test))
    print("min and max of y_train     : ", np.min(y_train), np.max(y_train))
    print("min and max of y_test      : ", np.min(y_test), np.max(y_test))
    print()


def load_noisy_bowl():
    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    x_grid, y_grid = np.meshgrid(x, y)
    z0_grid = x_grid**2 + y_grid**2

    np.random.seed(1)
    ep = np.random.randn(x_grid.shape[0], x_grid.shape[1])
    z_grid = z0_grid + 0.3 * ep

    X = x_grid.reshape([-1, 1])
    Y = y_grid.reshape([-1, 1])
    Z = z_grid.reshape([-1, 1])

    data = np.hstack([X, Y, Z]).astype(np.float32)

    return data


if __name__ == '__main__':
    print("load_noisy_bowl")
    print()

    data = load_noisy_bowl()
    x_grid = data[:, 0].reshape(40, 40)
    y_grid = data[:, 1].reshape(40, 40)
    z_grid = data[:, 2].reshape(40, 40)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Noisy Bowl')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.0, 2.5])
    plt.show()


def load_noisy_bowl_for_linear_regression():
    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    x_grid, y_grid = np.meshgrid(x, y)
    z0_grid = x_grid**2 + y_grid**2

    np.random.seed(1)
    ep = np.random.randn(x_grid.shape[0], x_grid.shape[1])
    z_grid = z0_grid + 0.3 * ep

    X = x_grid.reshape([-1, 1])
    Y = y_grid.reshape([-1, 1])
    Z = z_grid.reshape([-1, 1])

    x_train = np.hstack([X, Y]).astype(np.float32)
    y_train = Z.astype(np.float32)

    data = (x_train, y_train)

    return data


if __name__ == '__main__':
    print("load_noisy_bowl_for_linear_regression")
    print()

    x_train, y_train = load_noisy_bowl_for_linear_regression()
    x_grid = x_train[:, 0].reshape(40, 40)
    y_grid = x_train[:, 1].reshape(40, 40)
    z_grid = y_train.reshape(40, 40)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Noisy Bowl')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.0, 2.5])
    plt.show()


def load_inception_model():
    # make directory if not exist
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/CIFAR-10"):
        os.mkdir("data/CIFAR-10")
    if not os.path.isdir("inception"):
        os.mkdir("inception")
    if not os.path.isdir("img"):
        os.mkdir("img")
    if not os.path.isdir("graphs"):
        os.mkdir("graphs")

    # load inception model
    inception.maybe_download()  # download inception data (85MB) if not exist in inception directory
    model = inception.Inception()  # load inception model

    return model


if __name__ == '__main__':
    print("load_inception_model")
    print()

    def classify(image_path):
        """ display image in image_path,
            classify using inception model,
            and print top 10 predictions """
        display(Image(image_path))
        plt.close('all')  # display image in image_path
        pred = model.classify(image_path=image_path)  # classify using inception model
        model.print_scores(pred=pred, k=10, only_first_name=True)  # print top 10 predictions

    def plot_resized_image(image_path):
        """ get resized image by putting image in image_path into inception model,
            and plot resized image """
        resized_image = model.get_resized_image(image_path=image_path)  # get resized image
        plt.imshow(resized_image, interpolation='nearest')
        plt.show()
        plt.close('all')  # plot resized image

    model = load_inception_model()

    print("Display, Classify, and Print Top 10 Predictions")
    classify(image_path="images/cropped_panda.jpg")
    plt.close('all')
    classify(image_path="images/parrot.jpg")
    plt.close('all')

    print("Plot Resized Images")
    plot_resized_image(image_path="images/parrot.jpg")
    plt.close('all')
    plot_resized_image(image_path="images/elon_musk_100x100.jpg")
    plt.close('all')


def load_cifar10_transfer_values():
    # load inception model
    model = load_inception_model()

    # load cifar10 dataset
    data = load_cifar10()
    x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names = data

    # compute, cache, and read transfer-values
    data_dir = "data/CIFAR-10/"
    file_path_cache_train = os.path.join(data_dir, 'inception_cifar10_train.pkl')
    file_path_cache_test = os.path.join(data_dir, 'inception_cifar10_test.pkl')

    print("Processing Inception transfer-values for training-images ...")
    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = x_train * 255.0
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    x_train_transfer_values = transfer_values_cache(
        cache_path=file_path_cache_train,
        images=images_scaled,
        model=model)

    print("Processing Inception transfer-values for test-images ...")
    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = x_test * 255.0
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    x_test_transfer_values = transfer_values_cache(
        cache_path=file_path_cache_test,
        images=images_scaled,
        model=model)

    data = (x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names, x_train_transfer_values, x_test_transfer_values)

    return data


if __name__ == '__main__':
    print("load_cifar10_transfer_values")
    print()

    data = load_cifar10_transfer_values()

    x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names, x_train_transfer_values, x_test_transfer_values = data

    print("x_train.shape                 : ", x_train.shape)
    print("x_train_transfer_values.shape : ", x_train_transfer_values.shape)
    print("x_test.shape                  : ", x_test.shape)
    print("x_test_transfer_values.shape  : ", x_test_transfer_values.shape)
    print("y_train.shape                 : ", y_train.shape)
    print("y_test.shape                  : ", y_test.shape)
    print("y_train_cls.shape             : ", y_train_cls.shape)
    print("y_test_cls.shape              : ", y_test_cls.shape)
    print()

    print("x_train.dtype                 : ", x_train.dtype)
    print("x_train_transfer_values.dtype : ", x_train_transfer_values.dtype)
    print("x_test.dtype                  : ", x_test.dtype)
    print("x_test_transfer_values.dtype  : ", x_test_transfer_values.dtype)
    print("y_train.dtype                 : ", y_train.dtype)
    print("y_test.dtypee                 : ", y_test.dtype)
    print("y_train_cls.dtype             : ", y_train_cls.dtype)
    print("y_test_cls.dtype              : ", y_test_cls.dtype)
    print()

    print("type(x_train)                 : ", type(x_train))
    print("type(x_train_transfer_values) : ", type(x_train_transfer_values))
    print("type(x_test)                  : ", type(x_test))
    print("type(x_test_transfer_values)  : ", type(x_test_transfer_values))
    print("type(y_train)                 : ", type(y_train))
    print("type(y_test)                  : ", type(y_test))
    print("type(y_train_cls)             : ", type(y_train_cls))
    print("type(y_test_cls)              : ", type(y_test_cls))
    print()

    print("min and max of x_train                 : ", np.min(x_train), np.max(x_train))
    print("min and max of x_train_transfer_values : ", np.min(x_train_transfer_values), np.max(x_train_transfer_values))
    print("min and max of x_test                  : ", np.min(x_test), np.max(x_test))
    print("min and max of x_test_transfer_values  : ", np.min(x_test_transfer_values), np.max(x_test_transfer_values))
    print("min and max of y_train                 : ", np.min(y_train), np.max(y_train))
    print("min and max of y_test                  : ", np.min(y_test), np.max(y_test))
    print("min and max of y_train_cls             : ", np.min(y_train_cls), np.max(y_train_cls))
    print("min and max of y_test_cls              : ", np.min(y_test_cls), np.max(y_test_cls))
    print()

    print('cls_names')
    print(cls_names)
    print()


def load_knifey():
    knifey.maybe_download_and_extract()

    dataset = knifey.load()
    x_train, y_train_cls, y_train = dataset.get_training_set()
    x_test, y_test_cls, y_test = dataset.get_test_set()
    cls_names = dataset.class_names

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_train_cls = y_train_cls.astype(np.int32)
    y_test_cls = y_test_cls.astype(np.int32)

    data = (x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names)

    return data


if __name__ == '__main__':
    print("load_knifey")
    print()

    def load_images(image_paths):
        # Load the images from disk.
        images = [imread(path) for path in image_paths]

        # Convert to a numpy array and return it.
        return np.asarray(images)

    def plot_images(images, cls_true, cls_pred=None, smooth=True):
        assert len(images) == len(cls_true)

        # Create figure with sub-plots.
        fig, axes = plt.subplots(3, 3)

        # Adjust vertical spacing.
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        for i, ax in enumerate(axes.flat):
            # There may be less than 9 images, ensure it doesn't crash.
            if i < len(images):
                # Plot image.
                ax.imshow(images[i],
                          interpolation=interpolation)

                # Name of the true class.
                cls_true_name = cls_names[cls_true[i]]

                # Show true and predicted classes.
                if cls_pred is None:
                    xlabel = "True: {0}".format(cls_true_name)
                else:
                    # Name of the predicted class.
                    cls_pred_name = cls_names[cls_pred[i]]

                    xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

                # Show the classes as the label on the x-axis.
                ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()
        plt.close('all')

    data = load_knifey()

    x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names = data

    #print("x_train.shape     : ", x_train.shape)
    #print("x_test.shape      : ", x_test.shape)
    print("y_train.shape     : ", y_train.shape)
    print("y_test.shape      : ", y_test.shape)
    print("y_train_cls.shape : ", y_train_cls.shape)
    print("y_test_cls.shape  : ", y_test_cls.shape)
    print()

    #print("x_train.dtype     : ", x_train.dtype)
    #print("x_test.dtype      : ", x_test.dtype)
    print("y_train.dtype     : ", y_train.dtype)
    print("y_test.dtypee     : ", y_test.dtype)
    print("y_train_cls.dtype : ", y_train_cls.dtype)
    print("y_test_cls.dtype  : ", y_test_cls.dtype)
    print()

    print("type(x_train)     : ", type(x_train))
    print("type(x_test)      : ", type(x_test))
    print("type(y_train)     : ", type(y_train))
    print("type(y_test)      : ", type(y_test))
    print("type(y_train_cls) : ", type(y_train_cls))
    print("type(y_test_cls)  : ", type(y_test_cls))
    print()

    #print("min and max of x_train     : ", np.min(x_train), np.max(x_train))
    #print("min and max of x_test      : ", np.min(x_test), np.max(x_test))
    print("min and max of y_train     : ", np.min(y_train), np.max(y_train))
    print("min and max of y_test      : ", np.min(y_test), np.max(y_test))
    print("min and max of y_train_cls : ", np.min(y_train_cls), np.max(y_train_cls))
    print("min and max of y_test_cls  : ", np.min(y_test_cls), np.max(y_test_cls))
    print()

    print("x_train[0] is not a np.ndarray but a path of an image : ", x_train[0])
    print("x_test[0] is not a np.ndarray but a path of an image  : ", x_test[0])
    print()

    print("cls_names : ", cls_names)
    print()

    images = load_images(image_paths=x_test[0:9])
    cls_true = y_test_cls[0:9]
    plot_images(images=images, cls_true=cls_true, smooth=True)
    print("images.shape : ", images.shape)
    print()


def load_knifey_transfer_values():
    # load inception model
    model = load_inception_model()

    # load cifar10 dataset
    data = load_knifey()
    x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names = data

    # compute, cache, and read transfer-values
    data_dir = "data/knifey-spoony/"
    file_path_cache_train = os.path.join(data_dir, 'inception-knifey-train.pkl')
    file_path_cache_test = os.path.join(data_dir, 'inception-knifey-test.pkl')

    print("Processing Inception transfer-values for training-images ...")
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    x_train_transfer_values = transfer_values_cache(
        cache_path=file_path_cache_train,
        image_paths=x_train,
        model=model)

    print("Processing Inception transfer-values for test-images ...")
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    x_test_transfer_values = transfer_values_cache(
        cache_path=file_path_cache_test,
        image_paths=x_test,
        model=model)

    data = (x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names, x_train_transfer_values, x_test_transfer_values)

    return data


if __name__ == '__main__':
    print("Data Loading knifey Transfer Values")
    print()

    data = load_knifey_transfer_values()

    x_train, y_train_cls, y_train, x_test, y_test_cls, y_test, cls_names, x_train_transfer_values, x_test_transfer_values = data

    #print("x_train.shape                 : ", x_train.shape)
    print("x_train_transfer_values.shape : ", x_train_transfer_values.shape)
    #print("x_test.shape                  : ", x_test.shape)
    print("x_test_transfer_values.shape  : ", x_test_transfer_values.shape)
    print("y_train.shape                 : ", y_train.shape)
    print("y_test.shape                  : ", y_test.shape)
    print("y_train_cls.shape             : ", y_train_cls.shape)
    print("y_test_cls.shape              : ", y_test_cls.shape)
    print()

    #print("x_train.dtype                 : ", x_train.dtype)
    print("x_train_transfer_values.dtype : ", x_train_transfer_values.dtype)
    #print("x_test.dtype                  : ", x_test.dtype)
    print("x_test_transfer_values.dtype  : ", x_test_transfer_values.dtype)
    print("y_train.dtype                 : ", y_train.dtype)
    print("y_test.dtypee                 : ", y_test.dtype)
    print("y_train_cls.dtype             : ", y_train_cls.dtype)
    print("y_test_cls.dtype              : ", y_test_cls.dtype)
    print()

    print("type(x_train)                 : ", type(x_train))
    print("type(x_train_transfer_values) : ", type(x_train_transfer_values))
    print("type(x_test)                  : ", type(x_test))
    print("type(x_test_transfer_values)  : ", type(x_test_transfer_values))
    print("type(y_train)                 : ", type(y_train))
    print("type(y_test)                  : ", type(y_test))
    print("type(y_train_cls)             : ", type(y_train_cls))
    print("type(y_test_cls)              : ", type(y_test_cls))
    print()

    #print("min and max of x_train                 : ", np.min(x_train), np.max(x_train))
    print("min and max of x_train_transfer_values : ", np.min(x_train_transfer_values), np.max(x_train_transfer_values))
    #print("min and max of x_test                  : ", np.min(x_test), np.max(x_test))
    print("min and max of x_test_transfer_values  : ", np.min(x_test_transfer_values), np.max(x_test_transfer_values))
    print("min and max of y_train                 : ", np.min(y_train), np.max(y_train))
    print("min and max of y_test                  : ", np.min(y_test), np.max(y_test))
    print("min and max of y_train_cls             : ", np.min(y_train_cls), np.max(y_train_cls))
    print("min and max of y_test_cls              : ", np.min(y_test_cls), np.max(y_test_cls))
    print()

    print("x_train[0] is not a np.ndarray but a path of an image : ", x_train[0])
    print("x_test[0] is not a np.ndarray but a path of an image  : ", x_test[0])
    print()

    print('cls_names')
    print(cls_names)
    print()
