import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from selectRegion import roipoly
from visualizeVocabulary import get_desc_imname
from visualizeVocabulary import get_keypoints

FRAME_DIR = './'
SIFT_DIR = './'


def gen_regions(query_imnames):
    for query_imname in query_imnames:
        query_im = Image.open(os.path.join(FRAME_DIR, query_imname))
        desc, _ = get_desc_imname(query_imname + ".mat")
        pos, _, _ = get_keypoints(query_imname + ".mat")

        fig, ax = plt.subplots()
        ax.imshow(query_im)
        roi_plotter = roipoly(roicolor='r')

        fig, ax = plt.subplots()
        ax.imshow(query_im)
        roi_plotter.displayROI()
        plt.savefig(query_imname + "_reg_query.png")
        plt.show()

        indices = roi_plotter.getIdx(query_im, pos)
        np.save(query_imname + '_reg_ind', indices)


if __name__ == '__main__':
    query_imnames = ['friends_0000000367.jpeg',
                     'friends_0000000527.jpeg',
                     'friends_0000000408.jpeg',
                     'friends_0000000112.jpeg',
                     'friends_0000000120.jpeg']
    gen_regions(query_imnames)
