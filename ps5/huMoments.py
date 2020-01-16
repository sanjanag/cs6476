import numpy as np
import pickle as pkl

def get_moment(I, p, q):
    h, w = I.shape
    xv = np.arange(1, h + 1).reshape(h, 1) + np.zeros((1, w), 'int')
    yv = np.arange(1, w + 1) + np.zeros((h, 1), 'int')
    return (np.power(xv, p) * np.power(yv, q) * I).sum()


def get_centroid(H):
    M00 = get_moment(H, 0, 0)
    return get_moment(H, 1, 0) / M00, get_moment(H, 0, 1) / M00


def get_central_moment(H, p, q):
    xbar, ybar = get_centroid(H)
    h, w = H.shape
    xv = np.arange(1, h + 1).reshape(h, 1) + np.zeros((1, w), 'int')
    yv = np.arange(1, w + 1) + np.zeros((h, 1), 'int')
    xv = xv - xbar
    yv = yv - ybar
    return (np.power(xv, p) * np.power(yv, q) * H).sum()


def huMoments(H):
    mu20 = get_central_moment(H, 2, 0)
    mu02 = get_central_moment(H, 0, 2)
    mu11 = get_central_moment(H, 1, 1)
    mu30 = get_central_moment(H, 3, 0)
    mu03 = get_central_moment(H, 0, 3)
    mu21 = get_central_moment(H, 2, 1)
    mu12 = get_central_moment(H, 1, 2)

    h1 = mu20 + mu02
    h2 = (mu20 - mu02) ** 2 + 4 * (mu11 ** 2)
    h3 = (mu30 - 3 * mu12) ** 2 + (3 * mu21 - mu03) ** 2
    h4 = (mu30 + mu12) ** 2 + (mu21 + mu03) ** 2
    h5 = (mu30 - 3 * mu12) * (mu30 + mu12) * (
            (mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) + (
                 3 * mu21 - mu03) * (mu21 + mu03) * (
                 3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)
    h6 = (mu20 - mu02) * (
            (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2) + 4 * mu11 * (
                 mu30 + mu12) * (mu21 + mu03)
    h7 = (3 * mu21 - mu03) * (mu30 + mu12) * (
            (mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) - (
                 mu30 - 3 * mu12) * (mu21 + mu03) * (
                 3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)
    feature = np.array([h1, h2, h3, h4, h5, h6, h7])
    return feature

if __name__ == '__main__':
    with open("allMHIs.npy", "rb") as f:
        allMHIs = pkl.load(f)
    num_seq = allMHIs.shape[-1]
    huVectors = []
    for i in range(num_seq):
        huVectors.append(huMoments(allMHIs[:,:,i]))
    huVectors = np.array(huVectors)
    with open("huVectors.npy", "wb") as f:
        pkl.dump(huVectors, f)
