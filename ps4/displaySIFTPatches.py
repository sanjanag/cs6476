import matplotlib.pyplot as plt

from computeSIFTPatches import computeSIFTPatches


def displaySIFTPatches(im, positions, scales, orients):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    corners = computeSIFTPatches(positions, scales, orients)

    for j in range(len(corners)):
        ax.plot([corners[j][0][1], corners[j][1][1]],
                [corners[j][0][0], corners[j][1][0]], color='g', linestyle='-',
                linewidth=1)
        ax.plot([corners[j][1][1], corners[j][2][1]],
                [corners[j][1][0], corners[j][2][0]], color='g', linestyle='-',
                linewidth=1)
        ax.plot([corners[j][2][1], corners[j][3][1]],
                [corners[j][2][0], corners[j][3][0]], color='g', linestyle='-',
                linewidth=1)
        ax.plot([corners[j][3][1], corners[j][0][1]],
                [corners[j][3][0], corners[j][0][0]], color='g', linestyle='-',
                linewidth=1)
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(0, im.shape[0])
    plt.gca().invert_yaxis()
