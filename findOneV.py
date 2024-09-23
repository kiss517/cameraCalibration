import numpy as np


def compute_vectors(H, i, j):
    v1 = H[0, i-1] * H[0, j-1]
    v2 = H[0, i-1] * H[1, j-1] + H[1, i-1] * H[0, j-1]
    v3 = H[0, i-1] * H[2, j-1] + H[2, i-1] * H[0, j-1]
    v4 = H[1, i-1] * H[1, j-1]
    v5 = H[1, i-1] * H[2, j-1] + H[2, i-1] * H[1, j-1]
    v6 = H[2, i-1] * H[2, j-1]
    
    return [v1, v2, v3, v4, v5, v6]


def calculateOneV(H):
 vector1 = np.array(compute_vectors(H,1,2),dtype=np.float64)
 vector2 = np.array(compute_vectors(H,1,1),dtype=np.float64)
 vector3 = np.array(compute_vectors(H,2,2),dtype=np.float64)
 V = []
 V.append(vector1)
 V.append(vector2-vector3)

 return V


