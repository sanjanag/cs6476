import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from data_paths import allLabels, actions_dict
from predictAction import get_distances
from generateAllMHIs import display_mhi


def display_top_results(topMHIs, actions, test_action):
    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(2, 2),
                     share_all=True,
                     axes_pad=.5)

    for axes, i in zip(grid, range(topMHIs.shape[-1])):
        axes.set_title(f"Result {i + 1} Action: {actions[i]}",
                       fontdict=None,
                       loc='center',
                       color="k")
        axes.imshow(topMHIs[:, :, i], cmap='gray')
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])
    plt.title(test_action)
    plt.savefig(f"results_{test_action}.png")
    plt.show()


if __name__ == '__main__':
    K = 4

    with open("allMHIs.npy", "rb") as f:
        allMHIs = pkl.load(f)
    with open("huVectors.npy", "rb") as f:
        huVectors = pkl.load(f)

    num_seq = len(allLabels)

    input_indices = [12,18]

    for test_idx in input_indices:
        mask = np.arange(num_seq) != test_idx
        train_indices = np.arange(num_seq)[mask]

        test_moments = huVectors[test_idx]
        test_label = allLabels[test_idx]
        train_moments = huVectors[train_indices]
        train_labels = [allLabels[idx] for idx in train_indices]
        test_action = actions_dict[test_label]

        train_mhi = allMHIs[:, :, train_indices]
        test_mhi = allMHIs[:, :, test_idx]

        distances = get_distances(test_moments, train_moments)
        topK = np.argsort(distances)[:K]
        topMHIs = train_mhi[:, :, topK]
        top_labels = [train_labels[idx] for idx in topK]
        top_actions = [actions_dict[label] for label in top_labels]

        display_mhi(test_mhi, test_action)
        display_top_results(topMHIs, top_actions, test_action)