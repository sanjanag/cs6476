import numpy as np
from PIL import Image
from computeH import computeH
import matplotlib.pyplot as plt


def rectify(inputIm, points, nw, nh):
    hi, wi = inputIm.shape[:2]
    H = computeH(points, np.array([[0, 0], [nw, 0], [nw, nh], [0, nh]]).T)
    hinv = np.linalg.inv(H)

    bbox = np.array([[0, 0, wi, wi], [0, hi, 0, hi], [1, 1, 1, 1]])
    warped_bbox = np.matmul(H, bbox)
    warped_bbox = warped_bbox / warped_bbox[-1]

    xmin = int(warped_bbox[0].min())
    xmax = int(warped_bbox[0].max())
    ymin = int(warped_bbox[1].min())
    ymax = int(warped_bbox[1].max())
    width = xmax - xmin
    height = ymax - ymin

    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

    ref_coord = np.vstack((xv.ravel(), yv.ravel(), np.ones(xv.size)))
    inp_coord = np.matmul(hinv, ref_coord)
    inp_coord = inp_coord / inp_coord[-1]
    inp_x = (inp_coord[0]).astype('int')
    inp_y = (inp_coord[1]).astype('int')

    valid = (inp_x < wi) & (inp_x >= 0) & (inp_y >= 0) & (inp_y < hi)
    warp_im = np.zeros((width * height, 3), dtype='uint8')
    for i in range(width * height):
        if valid[i]:
            warp_im[i, :] = inputIm[inp_y[i], inp_x[i], :]
    warp_im = warp_im.reshape((height, width, 3))

    offset_x = 0
    offset_y = 0
    if ymin < 0:
        offset_y = int(abs(ymin))
    if xmin < 0:
        offset_x = int(abs(xmin))

    warp_im = warp_im[offset_y:offset_y + nh, offset_x:offset_x + nw, :]

    return warp_im

def run(t1_file, inp_img_file, width, height, name):
    t1 = np.load(t1_file).T
    inp_img = np.asarray(Image.open(inp_img_file))

    fig, ax = plt.subplots()
    ax = plt.imshow(inp_img)
    plt.scatter(t1[0], t1[1], marker='+', zorder=2, color='red')
    plt.savefig('inp_' + name + '.jpg', dpi=200)

    wim = rectify(inp_img, t1, width, height)

    Image.fromarray(wim).save('warp_' + name + '.png')

if __name__ == "__main__":
   run('./bldpoints.npy', './building.jpg', 700, 500, 'building')