import numpy as np

def calculateRT(H, K):
    K_1 = np.linalg.inv(K)
    
    # 提取H的三列
    H1 = H[:, 0].reshape(-1, 1)
    H2 = H[:, 1].reshape(-1, 1)
    H3 = H[:, 2].reshape(-1, 1)
    
    # 计算旋转向量和平移向量
    r1 = np.dot(K_1, H1).reshape(-1, 1)
    r2 = np.dot(K_1, H2).reshape(-1, 1)
    t = np.dot(K_1, H3).reshape(-1, 1)
    
    # 计算尺度因子，使用r1的尺度因子
    s = 1 / np.linalg.norm(r1)
    
    # 缩放旋转向量和平移向量
    r1 = r1 * s
    r2 = r2 * s
    t = t * s
    
    # 计算r3
    r3 = np.cross(r1.reshape(3), r2.reshape(3)).reshape(-1, 1)
    
    # 构造旋转矩阵R并正交化
    R = np.hstack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    
    # 构造RT矩阵
    RT = np.hstack((R, t))
    RT = np.vstack((RT, np.array([0, 0, 0, 1])))
    
    return RT, s


# 示例数据
# H = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# K = np.array([[1, 0, 0],
#               [0, 1, 0],
#               [0, 0, 1]])

# RT, s = calculateRT(H, K)

# # 打印结果
# print("RT矩阵:")
# print(RT)
# print("尺度因子:")
# print(s)

# # 检查 K * RT 是否等于 H
# H_reconstructed = np.dot(K, rt)[:3, :3]
# print("重构的H矩阵:")
# print(H_reconstructed)
# print("原始的H矩阵:")
# print(H)
# print("误差:")
# print(H - H_reconstructed)
