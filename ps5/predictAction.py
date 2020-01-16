import pickle as pkl

import numpy as np

from data_paths import allLabels


def get_distances(test_moments, train_moments):
    variance = np.var(train_moments, axis=0)
    distances = np.sqrt(
        np.sum(np.power(test_moments - train_moments, 2) / variance, axis=1))
    return distances


def predictAction(test_moments, train_moments, trainLabels):
    distances = get_distances(test_moments, train_moments)
    matched = np.argsort(distances)[0]
    return [trainLabels[matched]]


if __name__ == '__main__':
    with open("huVectors.npy", "rb") as f:
        huVectors = pkl.load(f)
    num_seq = len(allLabels)

    test_idx = 0
    mask = np.arange(num_seq) != test_idx
    train_indices = np.arange(num_seq)[mask]

    test_moments = huVectors[test_idx]
    train_moments = huVectors[train_indices]
    train_labels = [allLabels[i] for i in train_indices]

    print(predictAction(test_moments, train_moments, train_labels))
