import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_one_image(image, img_shape, cls=None, cls_pred=None):
    plt.figure()
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    if (cls is not None) and (cls_pred is not None):
        xlabel = "True: {0}, Pred: {1}".format(cls, cls_pred)
    elif (cls is not None) and (cls_pred is None):
        xlabel = "True: {0}".format(cls)
    elif (cls is None) and (cls_pred is not None):
        xlabel = "Pred: {0}".format(cls_pred)
    else:
        xlabel = ""
    plt.xlabel(xlabel)

    plt.show()
    plt.close()


def plot_many_images_2d(images, img_shape, cls=None, cls_pred=None):
    n = len(images)
    sqrt_n = int(np.ceil(np.sqrt(n)))

    plt.figure()
    fig, axes = plt.subplots(sqrt_n, sqrt_n)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            if (cls is not None) and (cls_pred is not None):
                xlabel = "True: {0}\nPred: {1}".format(cls[i], cls_pred[i])
            elif (cls is not None) and (cls_pred is None):
                xlabel = "True: {0}".format(cls[i])
            elif (cls is None) and (cls_pred is not None):
                xlabel = "Pred: {0}".format(cls_pred[i])
            else:
                xlabel = ""
            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.close()


def plot_many_images_3d(images, cls=None, cls_pred=None):
    n = images.shape[3]
    sqrt_n = int(np.ceil(np.sqrt(n)))

    plt.figure()
    fig, axes = plt.subplots(sqrt_n, sqrt_n)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[0, :, :, i], cmap='binary')

            if (cls is not None) and (cls_pred is not None):
                xlabel = "True: {0}, Pred: {1}".format(cls[i], cls_pred[i])
            elif (cls is not None) and (cls_pred is None):
                xlabel = "True: {0}".format(cls[i])
            elif (cls is None) and (cls_pred is not None):
                xlabel = "Pred: {0}".format(cls_pred[i])
            else:
                xlabel = ""
            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.close()


def plot_16_images_2d_and_return(images, img_shape):

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image.reshape(img_shape), cmap='Greys_r')
    plt.show()
    plt.close("all")

    return fig
