import numpy as np
def findHomography(objp, imgp):
    # print(f"Number of objpoints: {len(objp)}")
    # print(f"Number of imgpoints: {len(imgp)}")
    assert len(objp) == len(imgp), "The number of object points and corner points must be the same."
    assert len(objp) >= 4, "At least 4 points are required to compute the homography."
    num_points = len(objp)
    A = []
    b=[]
    for i in range(num_points):
        x_w, y_w = objp[i][0], objp[i][1]
        u, v = imgp[i][0], imgp[i][1] 
        A.append([x_w, y_w, 1, 0, 0, 0, -u * x_w, -u * y_w])
        A.append([0, 0, 0, x_w, y_w, 1, -v * x_w, -v * y_w])
        b.append(u)
        b.append(v)
    A = np.array(A,dtype=np.float64)
    b = np.array(b).reshape(-1,1)
    # print(A.shape)
    # print(b.shape)
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # 计算 Σ 的伪逆
    S_inv = np.diag(1 / S)
    # 计算 A 的伪逆
    A_pseudo_inv = VT.T @ S_inv @ U.T
    # 计算最小二乘解
    x_ls = A_pseudo_inv @ b    
    H = np.zeros((3, 3))
    H.flat[:8] = x_ls


    # 最后一个元素填1
    H[2, 2] = 1
    return H
