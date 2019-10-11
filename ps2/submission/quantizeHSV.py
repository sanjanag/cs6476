from skimage.color import rgb2hsv, hsv2rgb
from sklearn.cluster import KMeans


def quantizeHSV(origImg, k):
    h, w, d = origImg.shape
    hsv_img = rgb2hsv(origImg)
    hue_values = hsv_img[:, :, 0].reshape((h * w), 1)
    kmeans = KMeans(n_clusters=k).fit(hue_values)
    labels = kmeans.predict(hue_values).reshape((h, w))
    centers = kmeans.cluster_centers_
    clustered_img = hsv_img.copy()
    for i in range(h):
        for j in range(w):
            clustered_img[i, j, 0] = centers[labels[i, j]]
    return (hsv2rgb(clustered_img) * 255).astype('uint8'), centers
