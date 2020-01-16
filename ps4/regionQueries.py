import os
import pickle as pkl

import numpy as np
from PIL import Image

from util import display_result_grid
from util import query_database
from visualizeVocabulary import get_desc_imname

FRAME_DIR = './'
SIFT_DIR = './'
NUM_WORDS = 500
NUM_FRAMES = 100
NUM_DESC = 2000
SAMPLE_STEP = 5
M = 5


def get_vec(model, desc):
    labels = model.predict(desc)
    vec = np.histogram(labels, bins=np.arange(NUM_WORDS + 1))[0]
    return vec


if __name__ == '__main__':
    query_imnames = ['friends_0000000367.jpeg',
                     'friends_0000000527.jpeg',
                     'friends_0000000408.jpeg',
                     'friends_0000000120.jpeg']

    codebook_fname = "codebook_" + str(NUM_WORDS) + "_" + str(NUM_FRAMES) + \
                     ".npy"
    imnames = pkl.load(open("imnames", "rb"))
    codebook = np.load(codebook_fname)
    model = pkl.load(open("model.pkl", "rb"))

    for query_imname in query_imnames:
        query_im = Image.open(os.path.join(FRAME_DIR, query_imname))
        desc, _ = get_desc_imname(query_imname + ".mat")

        indices = np.load(query_imname + "_reg_ind.npy")

        desc = desc[indices]
        res_imnames = query_database(desc, model, imnames, codebook)
        display_result_grid(query_imname + "_reg_query.png", res_imnames,
                            "reg")
