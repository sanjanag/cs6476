import numpy as np


def find_optimal_vertical_seam(cumulativeEnergyMap):
    m, n = cumulativeEnergyMap.shape
    cols = [np.where(cumulativeEnergyMap[-1, :] == np.amin(cumulativeEnergyMap[-1, :]))[0][0]]
    for i in range(m - 2, -1, -1):
        min_energy = cumulativeEnergyMap[i, cols[-1]]
        idx = cols[-1]
        if cols[-1] + 1 < n and min_energy > cumulativeEnergyMap[i, cols[-1] + 1]:
            min_energy = cumulativeEnergyMap[i, cols[-1] + 1]
            idx = cols[-1] + 1
        if cols[-1] - 1 >= 0 and min_energy > cumulativeEnergyMap[i, cols[-1] - 1]:
            idx = cols[-1] - 1
        cols.append(idx)
    return cols[::-1]
