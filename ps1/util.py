import numpy as np
from PIL import Image
from scipy import ndimage

HOR_DIR = "HORIZONTAL"
VER_DIR = "VERTICAL"


def energy_image(im):
    gray_im = np.asarray(Image.fromarray(im).convert('L'))
    sx = ndimage.sobel(gray_im, axis=0, output='double', mode='mirror')
    sy = ndimage.sobel(gray_im, axis=1, output='double', mode='mirror')
    return np.add(np.absolute(sx), np.absolute(sy))


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


def find_optimal_horizontal_seam(cumulativeEnergyMap):
    m, n = cumulativeEnergyMap.shape
    rows = [np.where(cumulativeEnergyMap[:, -1] == np.amin(cumulativeEnergyMap[:, -1]))[0][0]]
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


def reduceWidth(im, energyImage):
    cols = find_optimal_vertical_seam(cumulative_minimum_energy_map(energyImage, VER_DIR))
    m, n, d = im.shape
    mask = np.ones(im.shape, dtype=bool)
    mask[np.arange(m), cols, :] = False
    reducedColorImage = im[mask].reshape((m, -1, d))
    reducedEnergyImage = energy_image(reducedColorImage)
    return reducedColorImage, reducedEnergyImage


def reduceHeight(im, energyImage):
    rows = find_optimal_horizontal_seam(cumulative_minimum_energy_map(energyImage, HOR_DIR))
    m, n, d = im.shape
    mask = np.ones(im.shape, dtype=bool)
    mask[rows, np.arange(n), :] = False
    reducedColorImage = im[mask].reshape((-1, n, d))
    reducedEnergyImage = energy_image(reducedColorImage)
    return reducedColorImage, reducedEnergyImage


def displaySeam(im, seam, type):
    pass
