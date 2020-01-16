import os
import pickle as pkl

import numpy as np
from PIL import Image

from util import display_result_grid
from util import query_database
from visualizeVocabulary import get_desc_imname
from visualizeVocabulary import get_keypoints

FRAME_DIR = './'
SIFT_DIR = './'
NUM_WORDS = 500
NUM_FRAMES = 100
NUM_DESC = 2000
SAMPLE_STEP = 5
M = 5

if __name__ == '__main__':
    query_imnames = ['friends_0000000070.jpeg',
                     'friends_0000000171.jpeg',
                     'friends_0000000257.jpeg']

    codebook_fname = "codebook_" + str(NUM_WORDS) + "_" + str(NUM_FRAMES) + \
                     ".npy"
    imnames = pkl.load(open("imnames", "rb"))
    codebook = np.load(codebook_fname)
    model = pkl.load(open("model.pkl", "rb"))

    for query_imname in query_imnames:
        query_im = Image.open(os.path.join(FRAME_DIR, query_imname))
        desc, _ = get_desc_imname(query_imname + ".mat")
        res_imnames = query_database(desc, model, imnames, codebook)
        display_result_grid(query_imname, res_imnames, "reg")
