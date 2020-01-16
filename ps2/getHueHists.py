import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans


def getHueHists(im, k):
    hsv_img = rgb2hsv(im)
    hue_values = hsv_img[:, :, 0].reshape((-1, 1))
    h_equal = np.histogram(hue_values, bins=k)
    c1, b1 = h_equal
    plt.figure(dpi=100)
    plt.hist(b1[:-1], b1, weights=c1, ec='k')
    plt.xlabel('Hue values')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of equally-spaces bins')
    for i in range(len(b1[:-1])):
        plt.text(b1[i], c1[i] + 1000, str(c1[i]))
    plt.show()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(hue_values)
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(hue_values)
    h_cluster = np.histogram(labels, bins=k, range=(0,k))
    c2, b2 = h_cluster
    plt.figure(dpi=100)
    plt.hist(b2[:-1], b2, weights=c2, ec='k')
    for i in range(len(b2[:-1])):
        plt.text(b2[i], c2[i] + 900, str(round(centers[i][0],2)))
    plt.xlabel('Labels')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of Clusters')
    plt.show()
    return h_equal, h_cluster