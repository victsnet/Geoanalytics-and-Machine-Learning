import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def grid(df, variables_rgb, shape, x_coordinates, y_coordinates, stretch=2, alpha=1.0):
    '''
    df - Pandas Dataframe.
    variables_rgb - list of the three variables chosen to create a RGB plot, in this order. Ex.: ['R', 'G', 'B'].
    shape - list with the grid's shape, e.g., [yshape, xshape].
    x_coordinates - X coordinates as Pandas Series.
    y_coordinates - Y coordinates as Pandas Series.
    stretch - the greater, the more points will be in extreme colors (black and white); stretch = 2 by default.
    alpha - controls the plot's transparency; alpha = 1.0 by default.

    '''
    import warnings

    if stretch <= 0:
        # displaying the warning message
        warnings.warn('Warning Message: Stretch must be greater than zero!')

    if (stretch < 1 and stretch > 0):
        # filtering the warning message
        warnings.filterwarnings('ignore')
          
    df = df
    stretch = stretch
    alpha = alpha
    shape = shape

    rgb = df[variables_rgb].values

    for i in range(3):
        rgb_ = rgb[:, i]
        rgb_ = (rgb_ - np.mean(rgb_)) / np.std(rgb_)
        rgb[:, i] = (rgb_ / int(stretch)) + 0.5

    for i in range(3):
        irgb = rgb[:, i] > (rgb[:, i].mean() + rgb[:, i].std())
        rgb[:, i][irgb] = 1

        irgb = rgb[:, i] < (rgb[:, i].mean() - rgb[:, i].std())
        rgb[:, i][irgb] = 0

    real_x = np.array(x_coordinates)
    real_y = np.array(y_coordinates)

    rgb = rgb.reshape(shape[0], shape[1], 3)

    dx = (real_x[1] - real_x[0]) / 2.
    dy = (real_y[1] - real_y[0]) / 2.
    extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

    return rgb, extent


def plot(rgb, variables_rgb, extent, triangle='on', figsize=[14, 10], triangle_position=[-0.0019, 0.5, .2, .2]):
    '''
    rgb - output from ternary function.
    variables_rgb -list of the three variables chosen to create a RGB plot, in this order. Ex.: ['R', 'G', 'B'].
    triangle - activate or desactivate RGB triangle ('on' by default or 'off').
    triangle_position - define the position fo the RGB triangle; default: [-0.02, 0.5, .2, .2].
    '''

    # maxwell triangle
    def colorTriangle(r, g, b):
        image = np.stack([r, g, b], axis=2)
        return image / image.max(axis=2)[:, :, None]

    size = 200
    X, Y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    u = np.full_like(X, .2)
    v = Y ** 1.5
    w = X ** 1.5

    ct = colorTriangle(u, v, w)

    indexes = np.arange(0, size, 1)
    matrix = np.zeros((size, size))
    ones = np.arange(1, size + 1, 1)[::-1]

    # cutting figure to triangle
    for i, n in zip(indexes, ones):
        matrix[i, :n] = 1

        matrix_symm = matrix[::-1]
        matrix_symm = matrix_symm.T[::-1]

    for k in range(ct.shape[2]):
        ct[:, :, k] = (ct[:, :, k] * matrix) + matrix_symm

    # plotting the rgb map and the maxwell triangle
    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.imshow(rgb, extent=extent)
    plt.grid(False)

    if triangle == 'on':
        ct[ct > 1] = 1
        a = plt.axes(triangle_position, facecolor='y')
        plt.imshow(ct)
        plt.axis('off')
        plt.grid(False)
        plt.text(-35, -6, variables_rgb[0], fontsize=12, weight='book')
        plt.text(-25, 215, variables_rgb[1], fontsize=12, weight='book')
        plt.text(200, -5, variables_rgb[2], fontsize=12, weight='book')

    if triangle == 'off':
        pass
