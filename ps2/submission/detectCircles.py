import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters


def get_grad_dir(img):
    sobelx = filters.sobel_h(img)
    sobely = filters.sobel_v(img)
    grad_dir = np.arctan2(sobely, sobelx)
    return grad_dir


def get_edges(img):
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img, ret * 0.5, ret, True)
    return edges


def get_sin_cos(theta_vals):
    sin_theta = {}
    cos_theta = {}
    for theta in theta_vals:
        sin_theta[theta] = np.sin(theta)
        cos_theta[theta] = np.cos(theta)
    return sin_theta, cos_theta


def detectCircles(im, radius, useGradient):
    #     gray_im = cv2.GaussianBlur(gray_im, (5,5),0)
    gray_im = np.asarray(Image.fromarray(im).convert('L'))

    # detect edges
    edges = get_edges(gray_im)

    # theta range
    theta_res = 1
    theta_vals = np.arange(-180, 180, theta_res) * np.pi / 180
    sin, cos = get_sin_cos(theta_vals)

    # grad dir ranging from (-pi,pi)
    grad_dir = get_grad_dir(gray_im)

    # construct accummulator array
    m, n = gray_im.shape
    acc = np.zeros(gray_im.shape, dtype='uint8')

    for i in range(m):
        for j in range(n):
            if edges[i, j] == 255:
                if useGradient:
                    theta = grad_dir[i, j]
                    b = int(round(j - radius * np.sin(theta)))
                    a = int(round(i - radius * np.cos(theta)))
                    if a >= 0 and a < m and b >= 0 and b < n:
                        acc[a][b] += 1
                else:
                    for theta in theta_vals:
                        b = int(round(j - radius * sin[theta]))
                        a = int(round(i - radius * cos[theta]))
                        if a >= 0 and a < m and b >= 0 and b < n:
                            acc[a][b] += 1

    # finding maxima
    x, y = np.where(acc > np.max(acc) * 0.8)
    return np.concatenate((y.reshape((-1, 1)), x.reshape((-1, 1))), axis=1)

# if __name__ == "__main__":
#     im = np.asarray(Image.open("./egg.jpg"))
#     print(detectCircles(im, 8.5, True))