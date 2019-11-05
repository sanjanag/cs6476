import numpy as np


def computeH(t1, t2):
    L = []
    for i in range(t1.shape[1]):
        p1 = [t1[0][i], t1[1][i], 1]
        p2 = [t2[0][i], t2[1][i], 1]
        a1 = [-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
              p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]]
        a2 = [0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
              p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]]
        L.append(a1)
        L.append(a2)
    L = np.array(L)
    u, s, v = np.linalg.svd(L)
    h = np.reshape(v[8], (3, 3))
    h = (1 / h[2][2]) * h
    return h
