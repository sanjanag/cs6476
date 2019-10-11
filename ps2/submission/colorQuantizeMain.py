import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists
from quantizeHSV import quantizeHSV
from quantizeRGB import quantizeRGB


def save_hist_equal(hist_equal, k):
    c1, b1 = hist_equal
    plt.figure(dpi=100)
    plt.hist(b1[:-1], b1, weights=c1, ec='k')
    plt.xlabel('Hue values')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of equally-spaces bins')
    for i in range(len(b1[:-1])):
        plt.text(b1[i], c1[i] + 1000, str(c1[i]))
    plt.savefig("hist_equal_" + str(k) + ".png", format="PNG")


def save_hist_clustered(hist_clustered, k):
    c2, b2 = hist_clustered
    plt.figure(dpi=100)
    plt.hist(b2[:-1], b2, weights=c2, ec='k')
    for i in range(len(b2[:-1])):
        plt.text(b2[i], c2[i] + 900, str(round(centers[i][0], 2)))
    plt.xlabel('Labels')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of Clusters')
    plt.savefig("hist_clustered_" + str(k) + ".png", format="PNG")


origImg = np.asarray(Image.open("fish.jpg"))

errors = []
k_range = [2, 5, 10, 20]

for k in k_range:
    quant_rgb, centers = quantizeRGB(origImg, k)
    errors.append(computeQuantizationError(origImg, quant_rgb))
    Image.fromarray(quant_rgb).save("quant_rgb_" + str(k) + ".png",
                                    format="PNG")

for k in k_range:
    quant_hsv, centers = quantizeHSV(origImg, k)
    errors.append(computeQuantizationError(origImg, quant_hsv))
    Image.fromarray(quant_hsv).save("quant_hsv_" + str(k) + ".png",
                                    format="PNG")

for k in k_range:
    hist_equal, hist_clustered = getHueHists(origImg, k)
    save_hist_equal(hist_equal, k)
    save_hist_clustered(hist_clustered, k)

print(errors)
