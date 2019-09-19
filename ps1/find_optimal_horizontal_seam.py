import numpy as np


def find_optimal_horizontal_seam(cumulativeEnergyMap):
    m, n = cumulativeEnergyMap.shape
    rows = [np.argmin(cumulativeEnergyMap[:,-1])]
    for i in range(n - 2, -1, -1):
        min_energy = cumulativeEnergyMap[rows[-1], i]
        idx = rows[-1]
        if rows[-1] + 1 < m and min_energy > cumulativeEnergyMap[rows[-1] + 1, i]:
            min_energy = cumulativeEnergyMap[rows[-1] + 1, i]
            idx = rows[-1] + 1
        if rows[-1] - 1 >= 0 and min_energy > cumulativeEnergyMap[rows[-1] - 1, i]:
            idx = rows[-1] - 1
        rows.append(idx)
    return rows[::-1]
