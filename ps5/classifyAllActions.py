import pickle as pkl

import numpy as np

from data_paths import allLabels, actions_dict
from predictAction import predictAction

if __name__ == '__main__':
    with open("huVectors.npy", "rb") as f:
        huVectors = pkl.load(f)

    num_seq = len(allLabels)
    num_actions = 5
    confusion_matrix = np.zeros((num_actions, num_actions), dtype='int')

    for test_idx in range(num_seq):
        mask = np.arange(num_seq) != test_idx
        train_indices = np.arange(num_seq)[mask]

        test_moments = huVectors[test_idx]
        test_label = allLabels[test_idx]
        train_moments = huVectors[train_indices]
        train_labels = [allLabels[idx] for idx in train_indices]

        predicted_label = predictAction(test_moments, train_moments,
                                      train_labels)[0]
        confusion_matrix[test_label-1][predicted_label-1] += 1

    print(confusion_matrix)
    orr = 0
    for i in range(5):
        orr += confusion_matrix[i][i]
    orr = orr / len(allLabels)
    clr = []
    for i in range(5):
        clr.append(confusion_matrix[i][i]/4)
    print(orr, clr)


