from find_optimal_horizontal_seam import *
from cumulative_minimum_energy_map import *
from energy_image import *


def reduceHeight(im, energyImage):
    rows = np.array(find_optimal_horizontal_seam(cumulative_minimum_energy_map(energyImage, HOR_DIR)))
    im = np.rot90(im, 1, (1, 0))
    m, n, d = im.shape
    mask = np.ones(im.shape, dtype=bool)
    mask[np.arange(m), n - rows, :] = False
    reducedColorImage = im[mask].reshape((m, n - 1, d))
    reducedColorImage = np.rot90(reducedColorImage, 1, (0, 1))
    reducedEnergyImage = energy_image(reducedColorImage)
    return reducedColorImage, reducedEnergyImage
