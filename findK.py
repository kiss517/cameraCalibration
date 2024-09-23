import math
import numpy as np


def findM_K1(B):
        # Cholesky 分解
    L = np.linalg.cholesky(B)
# 计算相机内参数
    L_inv = np.linalg.inv(L)

    f_x = 1 / L_inv[0, 0]
    f_y = 1 / L_inv[1, 1]
    s = -L_inv[1, 0] / L_inv[1, 1]
    u0 = L_inv[0, 2] / L_inv[0, 0]
    v0 = L_inv[1, 2] / L_inv[1, 1]

# 构建内参数矩阵 K
    K = np.array([
    [f_x, s, u0],
    [0, f_y, v0],
    [0, 0, 1]
     ],dtype=np.float64)

    print("相机内参数矩阵 K:")
    print(K)
    return K



def findM_K2(B):
    B1 = np.linalg.inv(B)
    B1/=B[2,2]
    u0 = B1[0, 2] / B1[2, 2]
    v0 = B1[1, 2] / B1[2, 2]
    f_y = math.sqrt(B1[1, 1] - v0 ** 2)
    s = (B1[0, 1] - u0 * v0) / f_y
    f_x = math.sqrt(B1[0, 0] - u0 ** 2 )
    print(s)

    K = np.array([
        [f_x, 0, u0],
        [0, f_y, v0],
        [0, 0, 1]
    ],dtype=np.float64)
    return K


def findM_K3(B):
    # 确保输入是一个 numpy 数组
    B = np.array(B)
    
    # 计算 v0
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1]**2)
    
    # 计算 temp
    temp = B[2, 2] - (B[0, 2]**2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    
    # 计算焦距 f_x 和 f_y
    f_x = math.sqrt(temp / B[0, 0])
    f_y = math.sqrt((temp * B[0, 0]) / (B[0, 0] * B[1, 1] - B[0, 1]**2))
    
    # 计算 s
    s = -B[0, 1] * f_x**2 * f_y / temp
    print(s)
    # 计算主点 u0
    u0 = s * v0 / f_y - B[0, 2] * f_x**2 / temp
    K = np.array([
        [f_x, s, u0],
        [0, f_y, v0],
        [0, 0, 1]
    ],dtype=np.float64)
    return K