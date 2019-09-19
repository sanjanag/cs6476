import numpy as np
from PIL import Image
from scipy import ndimage


def energy_image(im):
    gray_im = np.asarray(Image.fromarray(im).convert('L'))
    sx = ndimage.prewitt(gray_im, axis=0, output='double', mode='constant')
    sy = ndimage.prewitt(gray_im, axis=1, output='double', mode='constant')
    return np.add(np.absolute(sx), np.absolute(sy))
    # return ndimage.gaussian_laplace(gray_im, sigma=1)