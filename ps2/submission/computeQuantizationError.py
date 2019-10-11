import numpy as np


def computeQuantizationError(origImg, quantizedImg):
    a = origImg.copy().astype(np.int64)
    b = quantizedImg.copy().astype(np.int64)
    return np.sum((a - b) ** 2)
