import numpy as np
from sklearn.cluster import KMeans


def quantizeRGB(im, k):
    h, w, d = im.shape
    pixels = im.reshape((h * w, d))
    kmeans = KMeans(n_clusters=k).fit(pixels)
    labels = kmeans.predict(pixels).reshape((h, w))
    centers = kmeans.cluster_centers_
    quant_im = np.zeros((h, w, d), dtype='uint8')
    for i in range(h):
        for j in range(w):
            quant_im[i, j, :] = centers[labels[i, j]]
    return quant_im, centers.astype('int')
