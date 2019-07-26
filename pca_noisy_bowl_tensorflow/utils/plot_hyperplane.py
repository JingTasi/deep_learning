import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d


def plot_hyperplane(x, y, y_pred, intercept, slope):

    x0_min = np.min(x[:, 0])
    x0_max = np.max(x[:, 0])
    x1_min = np.min(x[:, 1])
    x1_max = np.max(x[:, 1])

    x0 = np.linspace(x0_min, x0_max)
    x1 = np.linspace(x1_min, x1_max)
    X0, X1 = np.meshgrid(x0, x1, indexing='xy')
    Y_pred = intercept[0] + X0 * slope[0][0] + X1 * slope[0][1]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X0, X1, Y_pred, alpha=0.2)
    ax.scatter3D(x[:, 0], x[:, 1], y.reshape(-1), c='k', s=100)
    ax.scatter3D(x[:, 0], x[:, 1], y_pred.reshape(-1), c='r', alpha=0.2)
    for i in range(len(y)):
        ax.plot([x[i, 0], x[i, 0]], [x[i, 1], x[i, 1]], [y[i, 0], y_pred[i, 0]], 'r', alpha=0.2)
    if not os.path.isdir('img'):
        os.mkdir('img')
    plt.savefig("img/linear regression with 2 features.png")
    plt.show()
