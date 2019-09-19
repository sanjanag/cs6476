import numpy as np
import matplotlib.pyplot as plt

HOR_DIR = "HORIZONTAL"
VER_DIR = "VERTICAL"


def displaySeam(im, seam, type):
    m, n, d = im.shape
    if type == HOR_DIR:
        im[seam, np.arange(n), 0] = 255
        im[seam, np.arange(n), 1:] = 0
        pass
    else:
        im[np.arange(m), seam, 0] = 255
        im[np.arange(m), seam, 1:] = 0
    plt.imshow(im)
