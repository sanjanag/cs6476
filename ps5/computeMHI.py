import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_mhi(mhi, action):
    plt.figure()
    plt.imshow(mhi, cmap='gray')
    plt.title(action)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(action + "_mhi.png")
    plt.show()



def computeMHI(dirpath):
    files = glob.glob(os.path.join(dirpath, "*.pgm"))
    files.sort()
    num_frames = len(files)
    thresh = 10
    frames = []
    for i in range(num_frames):
        frames.append(cv2.imread(files[i], 0))
    frames = np.array(frames, dtype='int')
    height, width = frames[0].shape
    D = np.zeros((num_frames, height, width), 'int')
    H = np.zeros((num_frames, height, width), 'int')
    for i in range(1, num_frames):
        D[i] = abs(frames[i] - frames[i - 1])
        mask = D[i] >= thresh
        H[i, mask] = num_frames
        H[i, ~mask] = H[i - 1, ~mask] - 1
        H[i] = np.maximum(H[i], 0)
    H = H[-1, :] / np.max(H)
    return H

if __name__ == '__main__':
    # show example MHIs
    example_mhis = ['./PS5_Data/botharms/botharms-up-p1-1/',
                    './PS5_Data/rightkick/rightkick-p1-1/',
                    './PS5_Data/crouch/crouch-p2-2/']
    actions = ['botharms', 'rightkick', 'crouch']
    for path, action in zip(example_mhis, actions):
        display_mhi(computeMHI(path), action)
