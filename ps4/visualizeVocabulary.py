import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

from dist2 import dist2
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from util import get_desc_imname
from util import get_keypoints

FRAME_DIR = './'
SIFT_DIR = './'
NUM_WORDS = 500
NUM_FRAMES = 100
NUM_DESC = 2000
SAMPLE_STEP = 5


def get_closes_patch(center, desc):
    distances = dist2(center.reshape(1, -1), desc)
    return np.argmin(distances)


if __name__ == "__main__":
    codebook_fname = "codebook_" + str(NUM_WORDS) + "_" + str(NUM_FRAMES) + \
                     ".npy"
    imnames = pkl.load(open("imnames", "rb"))
    codebook = np.load(codebook_fname)
    word_ids = [95, 401]

    for word_id in word_ids:
        indices = np.where(codebook[:, word_id] > 0)[0]

        model = pkl.load(open("model.pkl", "rb"))
        patches = []
        for idx in indices:
            imname = imnames[idx]
            sift_fname = imname + ".mat"
            gray_im = np.asarray(Image.open(os.path.join(FRAME_DIR,
                                                         imname)).convert('L'))

            desc, _ = get_desc_imname(sift_fname)
            pos, scale, orient = get_keypoints(sift_fname)

            closest_patch_idx = get_closes_patch(
                model.cluster_centers_[word_id],
                desc)
            # print(closest_patch_idx)
            patch = getPatchFromSIFTParameters(pos[closest_patch_idx], scale[
                closest_patch_idx], orient[closest_patch_idx], gray_im)
            patches.append(patch)

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(5, 5),
                         share_all=True,
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        count = 0
        for ax in grid:
            if count < len(patches):
                patch = patches[count]
                if patch.shape[0] > 40 or patch.shape[1] > 40:
                    patch_im = Image.fromarray(patch).resize((40, 40))
                else:
                    patch_im = Image.fromarray(patch)
                ax.imshow(np.asarray(patch_im), cmap='gray')
                count += 1
        plt.savefig("word_vis_" + str(word_id) + ".png")
        plt.show()
