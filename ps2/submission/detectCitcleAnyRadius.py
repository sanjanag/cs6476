import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from detectCircles import get_edges, get_sin_cos, get_grad_dir


def detectCircles(im, useGradient):
    radius_range = np.arange(5, 30, 0.5)
    gray_im = np.asarray(Image.fromarray(im).convert('L'))

    # detect edges
    edges = get_edges(gray_im)

    # theta range
    theta_res = 10
    theta_vals = np.arange(-180, 180, theta_res) * np.pi / 180
    sin, cos = get_sin_cos(theta_vals)

    # grad dir ranging from (-pi,pi)
    grad_dir = get_grad_dir(gray_im)

    # construct accummulator array
    m, n = gray_im.shape
    acc = np.zeros((m, n, radius_range.size), dtype='uint8')

    for i in range(m):
        for j in range(n):
            if edges[i, j] == 255:
                for k, radius in enumerate(radius_range):
                    if useGradient:
                        theta = grad_dir[i, j]
                        b = int(round(j - radius * np.sin(theta)))
                        a = int(round(i - radius * np.cos(theta)))
                        if 0 <= a < m and 0 <= b < n:
                            acc[a][b][k] += 1
                    else:
                        for theta in theta_vals:
                            b = int(round(j - radius * sin[theta]))
                            a = int(round(i - radius * cos[theta]))
                            if 0 <= a < m and 0 <= b < n:
                                acc[a][b][k] += 1

    # find centers
    x, y, k = np.where(acc > np.max(acc) * 0.8)

    # plot images
    fig, ax = plt.subplots(1, 1, True, True, figsize=(7, 7), dpi=100)
    ax.imshow(im, )

    def text(x, y, text):
        ax.text(x, y, text, ha='center', va='top', color='red')

    # plotting circles on top of image
    for i in range(x.shape[0]):
        circle = plt.Circle((y[i], x[i]), radius_range[k[i]], color='r',
                            fill=False)
        ax.add_patch(circle)
        text(y[i], x[i], str(radius_range[k[i]]))

    ax.title.set_text('Circles of any radius')
    return np.concatenate(
        (y.reshape((-1, 1)), x.reshape((-1, 1)), k.reshape((-1, 1))), axis=1)
