import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from displaySIFTPatches import displaySIFTPatches
from dist2 import dist2
from selectRegion import roipoly

if __name__ == '__main__':
    contents = sio.loadmat("./twoFrameData.mat")

    im1 = contents['im1']
    pos1 = contents['positions1']
    scales1 = contents['scales1']
    orients1 = contents['orients1']
    desc1 = contents['descriptors1']
    im2 = contents['im2']
    pos2 = contents['positions2']
    scales2 = contents['scales2']
    orients2 = contents['orients2']

    fig, ax = plt.subplots()
    ax.imshow(im1)
    roiplotter = roipoly(roicolor='r')
    fig, ax = plt.subplots()
    ax.imshow(im1)
    roiplotter.displayROI()
    plt.savefig("roi.png")
    plt.show()

    indices = roiplotter.getIdx(im1, pos1)
    # np.save('desc_chosen', indices)

    desc_chosen = desc1[indices]
    desc2 = contents['descriptors2']
    distances = dist2(desc_chosen, desc2)

    matches = np.argmin(distances, axis=1)

    displaySIFTPatches(im2, pos2[matches], scales2[matches], orients2[matches])
    plt.savefig("desc_match.png")
    plt.show()
