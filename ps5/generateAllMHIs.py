import pickle

import matplotlib.pyplot as plt
import numpy as np

from computeMHI import computeMHI
from data_paths import seqpaths


def display_mhi(mhi, action):
    plt.figure()
    plt.imshow(mhi, cmap='gray')
    plt.title(action)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(action + "_mhi.png")
    plt.show()


if __name__ == '__main__':
    # generate all MHIs
    allMHIs_list = []
    for path in seqpaths:
        H = computeMHI(path)
        allMHIs_list.append(H)

    allMHIs = np.stack(allMHIs_list, axis=-1)

    # dump MHIs
    with open('allMHIs.npy', "wb") as f:
        pickle.dump(allMHIs, f)
