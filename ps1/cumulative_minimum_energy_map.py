import numpy as np

HOR_DIR = "HORIZONTAL"
VER_DIR = "VERTICAL"


def cumulative_minimum_energy_map(energyImage, seamDirection):
    m, n = energyImage.shape
    cum_map = np.empty(energyImage.shape, dtype=float)
    if seamDirection == HOR_DIR:
        for i in range(m):
            cum_map[i, 0] = energyImage[i, 0]
        for j in range(1, n):
            for i in range(m):
                min_energy = cum_map[i, j - 1]
                if i - 1 >= 0:
                    min_energy = min(min_energy, cum_map[i - 1, j - 1])
                if i + 1 < m:
                    min_energy = min(min_energy, cum_map[i + 1, j - 1])
                cum_map[i, j] = energyImage[i, j] + min_energy
        return cum_map
    elif seamDirection == VER_DIR:
        for j in range(n):
            cum_map[0, j] = energyImage[0, j]
        for i in range(1, m):
            for j in range(n):
                min_energy = cum_map[i - 1, j]
                if j - 1 >= 0:
                    min_energy = min(min_energy, cum_map[i - 1, j - 1])
                if j + 1 < n:
                    min_energy = min(min_energy, cum_map[i - 1, j + 1])
                cum_map[i, j] = energyImage[i, j] + min_energy
        return cum_map