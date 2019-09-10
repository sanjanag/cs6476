from util import *

orig_im = np.asarray(Image.open("./inputSeamCarvingPrague.jpg"))
im = orig_im.copy()
count = 0
e_im = energy_image(im)
while count < 100:
    im, e_im = reduceWidth(im, e_im)
    count += 1
Image.fromarray(im).save("outputReduceWidthPrague.png", format="PNG")

orig_im = np.asarray(Image.open("./inputSeamCarvingMall.jpg"))
im = orig_im.copy()
count = 0
e_im = energy_image(im)
while count < 100:
    im, e_im = reduceWidth(im, e_im)
    count += 1
Image.fromarray(im).save("outputReduceWidthMall.png", format="PNG")

