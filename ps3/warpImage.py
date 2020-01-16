import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from computeH import computeH


def warpImage(inputIm, refIm, H):
    hi, wi = inputIm.shape[:2]
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

    img1 = Image.fromarray(refIm)
    img2 = Image.fromarray(warp_im)

    shift = (offset_x, offset_y)

    nw = max(offset_x + refIm.shape[1], warp_im.shape[1])
    nh = max(offset_y + refIm.shape[0], warp_im.shape[0])

    # m1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    m1 = Image.new('RGB', size=(nw, nh))
    m1.paste(img2, (0, 0))
    m1.paste(img1, shift)

    # m2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    m2 = Image.new('RGB', size=(nw, nh))
    m2.paste(img1, shift)
    m2.paste(img2, (0, 0))

    mosiac = Image.blend(m1, m2, alpha=0.5)
    return warp_im, np.asarray(mosiac)

def run(t1_file, t2_file, inp_img_file, ref_img_file, name):
    t1 = np.load(t1_file).T
    t2 = np.load(t2_file).T

    ref_img = np.asarray(Image.open(ref_img_file))
    inp_img = np.asarray(Image.open(inp_img_file))

    fig, ax = plt.subplots()
    ax = plt.imshow(inp_img)
    plt.scatter(t1[0], t1[1], marker='+', zorder=2, color='red')
    plt.savefig('inp_' + name + '.jpg', dpi=200)

    fig, ax = plt.subplots()
    ax = plt.imshow(ref_img)
    plt.scatter(t2[0], t2[1], marker='+', zorder=2, color='red')
    plt.savefig('ref_' + name + '.jpg', dpi=200)

    H = computeH(t1, t2)
    wim, mim = warpImage(inp_img, ref_img, H)

    Image.fromarray(wim).save('warp_' + name + '.png')

    Image.fromarray(mim).save('mosiac_' + name +  '.png')

if __name__ == "__main__":
   run('./cc1.npy', './cc2.npy', './crop1.jpg', './crop2.jpg', 'crop')
   run('./points1.npy', './points2.npy', './wdc1.jpg', './wdc2.jpg', 'wdc')
   run('./v1.npy', './v2.npy', './vinita1.jpg', './vinita2.jpg', 'vinita')