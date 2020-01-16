import glob
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from dist2 import dist2

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


def query_database(desc, model, imnames, codebook):
    vec = get_vec(model, desc)
    matches = np.argsort(cosine_similarity(vec.reshape(1, -1),
                                           codebook)[0])[::-1][:M]
    res_imnames = []
    for match in matches:
        res_imnames.append(imnames[match])
    # print(res_imnames)
    return res_imnames


def display_result_grid(query_imname, res_imnames, suffix):
    images = [Image.open(os.path.join(FRAME_DIR, query_imname))]
    for res_imname in res_imnames:
        images.append(Image.open(os.path.join(FRAME_DIR, res_imname)))

    fig = plt.figure(figsize=(10, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), share_all=True,
                     axes_pad=0.1)
    for ax, im in zip(grid, images):
        ax.imshow(im)
    plt.savefig(query_imname + "_" + suffix + ".png")
    plt.show()


def get_descriptors_bulk(fnames):
    desc_list = []
    imnames = []
    for fname in fnames:
        descriptors, imname = get_desc_imname(fname)
        if descriptors.shape[0] < 1:
            continue
        desc_list.append(descriptors[np.random.choice(
            descriptors.shape[0], min(NUM_DESC, descriptors.shape[0]), False),
                         :])
        imnames.append(imname)
    return desc_list, imnames


def get_desc_imname(fname):
    filepath = os.path.join(SIFT_DIR, fname)
    contents = sio.loadmat(filepath, verify_compressed_data_integrity=False)
    return contents['descriptors'], contents['imname'][0]


def get_keypoints(fname):
    filepath = os.path.join(SIFT_DIR, fname)
    contents = sio.loadmat(filepath, verify_compressed_data_integrity=False)
    return contents['positions'], contents['scales'], contents['orients']


def build_model(n_clusters, desc_list):
    features = np.vstack(desc_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1,
                    verbose=1).fit(features)
    return kmeans


def get_quant_vec(model, descriptors, n_words):
    labels = model.predict(descriptors)
    return np.histogram(labels, bins=np.arange(n_words + 1))[0]


def get_codebook(model, desc_list, n_words):
    vectors = []
    for desc in desc_list:
        vectors.append(get_quant_vec(model, desc, n_words))
    assert (len(vectors) == len(desc_list))
    return np.vstack(vectors).astype('int')


def get_sift_fnames(num_files):
    fnames = glob.glob(SIFT_DIR + '*.mat')
    fnames = [i[-27:] for i in fnames]
    fnames.sort()
    ret_fnames = []
    for i in range(len(fnames)):
        if i % SAMPLE_STEP == 0:
            ret_fnames.append(fnames[i])
    return ret_fnames[:num_files]


def build_codebook(num_words, num_frames):
    print("Reading sift filenames")
    sift_fnames = get_sift_fnames(num_frames)
    print("Reading descriptors")
    desc_list, imnames = get_descriptors_bulk(sift_fnames)
    print("Building model")
    model = build_model(NUM_WORDS, desc_list)
    print("Dumping model")
    pkl.dump(model, open("model.pkl", "wb"))
    print("Generating codebook")
    codebook = get_codebook(model, desc_list, NUM_WORDS)
    print("Dumping codebook")
    np.save('codebook_' + str(num_words) + '_' + str(num_frames), codebook)
    pkl.dump(imnames, open("imnames", "wb"))


def get_closes_patch(center, desc):
    distances = dist2(center.reshape(1, -1), desc)
    return np.argmin(distances)
