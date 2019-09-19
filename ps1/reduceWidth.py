from find_optimal_vertical_seam import *
from cumulative_minimum_energy_map import *
from energy_image import *


def reduceWidth(im, energyImage):
    cols = find_optimal_vertical_seam(cumulative_minimum_energy_map(energyImage, VER_DIR))
    m, n, d = im.shape
    mask = np.ones(im.shape, dtype=bool)
    mask[np.arange(m), cols, :] = False
    reducedColorImage = im[mask].reshape((m, n - 1, d))
    reducedEnergyImage = energy_image(reducedColorImage)
    return reducedColorImage, reducedEnergyImage
