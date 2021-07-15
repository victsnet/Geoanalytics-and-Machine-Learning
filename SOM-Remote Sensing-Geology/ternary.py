import warnings
import numpy as np
import matplotlib.pyplot as plt


def grid(df, variables_rgb, x_label, y_label, center=0.5, stretch=1.0, show_hist='on'):
    '''
    df - Pandas Dataframe.
    variables_rgb - list of the three variables chosen to create a RGB plot, in this order. Ex.: ['R', 'G', 'B'].
    x_label - X coordinates label.
    y_label - Y coordinates label.
    center - define the center of the distribution.
    stretch - to stretch distribution.
    '''

    nx = df[x_label].unique().size
    ny = df[y_label].unique().size
    shape = [ny, nx]
    rgb = df[variables_rgb].values
    rgb_base = {}
    
    for i in range(3):

        rgb_ = (rgb[:, i] - np.mean(rgb, axis=0)[i])/np.std(rgb, axis=0)[i]
        rgb_ = (rgb_/np.max(rgb_))/stretch
        dist = center - np.mean(rgb_)
        rgb_ = rgb_ + dist  # re-centering the distribution
        rgb_ = np.where(rgb_ <= 1, rgb_, 1)
        rgb_ = np.where(rgb_ >= 0, rgb_, 0)
        rgb_base[f'rgb_{i}'] = rgb_
        
    df = df.copy()    
    df[variables_rgb[0]] = rgb_base['rgb_0']
    df[variables_rgb[1]] = rgb_base['rgb_1']
    df[variables_rgb[2]] = rgb_base['rgb_2']
    rgb = df[variables_rgb].values
    

    if show_hist == 'on':
        
        fig, ax = plt.subplots(ncols=3, figsize=(16, 3)) 
        ax[0].hist(rgb_base['rgb_0'], bins=25, color='r', alpha=0.7)
        ax[0].set_title(variables_rgb[0])
        ax[0].grid(alpha=0.6)
        ax[1].hist(rgb_base['rgb_1'], bins=25, color='g', alpha=0.7)
        ax[1].set_title(variables_rgb[1])
        ax[1].grid(alpha=0.6)
        ax[2].hist(rgb_base['rgb_2'], bins=25, color='b', alpha=0.7)
        ax[2].set_title(variables_rgb[2])
        ax[2].grid(alpha=0.6)


    real_x = np.array(df[x_label])
    real_y = np.array(df[y_label])

    rgb = rgb.reshape(shape[0], shape[1], 3)[::-1]

    dx = (real_x[1] - real_x[0]) / 2.
    dy = (real_y[1] - real_y[0]) / 2.
    extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

    return rgb, extent

#rgb, variables_rgb, extent = grid()

def plot(rgb, variables_rgb, extent, triangle='on', triangle_position=[0.08, 0.2, .2, .2], alpha=1.0):
    '''
    rgb - output from ternary function.
    variables_rgb -list of the three variables chosen to create a RGB plot, in this order. Ex.: ['R', 'G', 'B'].
    triangle - activate or desactivate RGB triangle ('on' by default or 'off').
    triangle_position - define the position fo the RGB triangle; default: [0.08, 0.2, .2, .2].
    alpha - controls the plot's transparency; alpha = 1.0 by default.
    '''
    import warnings
    warnings.filterwarnings("ignore")

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

    # plotting rgb map and the maxwell triangle
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='w')
    plt.imshow(rgb, extent=extent, alpha=alpha)
    plt.grid(False)

    if triangle == 'on':
        ct[ct > 1] = 1
        a = plt.axes(triangle_position, facecolor='y')
        plt.imshow(ct)
        plt.axis('off')
        plt.grid(False)
        plt.text(-35.5, -6, variables_rgb[0], fontsize=12, weight='book')
        plt.text(-25, 215, variables_rgb[1], fontsize=12, weight='book')
        plt.text(200, -5, variables_rgb[2], fontsize=12, weight='book')

    if triangle == 'off':
        pass

    return ax