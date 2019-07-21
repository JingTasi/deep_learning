import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.image import cm

def plot_3d(x, title='Noisy Bowl', xlabel='X', ylabel='y', zlabel='z'):
    
    n = int(np.sqrt(x.shape[0]))
    
    x_grid = x[:, 0].reshape(n, n)
    y_grid = x[:, 1].reshape(n, n)
    z_grid = x[:, 2].reshape(n, n)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    xlim = np.max(np.absolute(x_grid))
    ylim = np.max(np.absolute(y_grid))
    zlim = np.max(np.absolute(z_grid))
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.set_zlim([-zlim, zlim])
    plt.show()