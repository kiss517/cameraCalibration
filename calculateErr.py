import numpy as np
import cv2
def project_points(K, objp, rvec, tvec, distortvec):

    rvec = np.reshape(rvec, (3, 1))
    tvec = np.reshape(tvec, (3, 1))
    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
  
    objp = np.array(objp).reshape(-1, 3).T #(3, N)

    # 进行矩阵乘法并加上位移向量
    img_prime = np.dot(R, objp) + tvec  # (3, N)

    
    # 归一化坐标
    x = img_prime[0] / img_prime[2]
    y = img_prime[1] / img_prime[2]
    # 畸变校正
    r_2 = x**2 + y**2
    x_distorted = x * (1 + distortvec[0] * r_2 + distortvec[1] * r_2**2 + distortvec[2] * r_2**3) + 2 * distortvec[3] * x * y + distortvec[4] * (r_2 + 2 * x**2)
    y_distorted = y * (1 + distortvec[0] * r_2 + distortvec[1] * r_2**2 + distortvec[2] * r_2**3) + distortvec[3] * (r_2 + 2 * y**2) + 2 * distortvec[4] * x * y


    # 将畸变校正后的点投影到图像平面
    u = x_distorted * K[0, 0] + K[0, 2]
    v = y_distorted * K[1, 1] + K[1, 2]

   
    # 将 u 和 v 结合成图像点
    imgpoints = np.column_stack((u, v))

    return imgpoints
def calculate_reprojection_error(params, objp, imgp):
    distortvec = params[:5]  # 提取畸变系数
    K = np.array([[params[5], 0, params[6]],
                  [0, params[7], params[8]],
                  [0, 0, 1]], dtype=np.float64)
    num_images = len(objp)  # 图像的数量
    total_error = []

    for i in range(num_images):
        # 提取旋转向量和位移向量
        rvec = params[9 + i * 6: 12 + i * 6]
        tvec = params[12 + i * 6: 15 + i * 6]
        
        # 计算投影点
        img_points_projected = project_points(K, objp[i], rvec, tvec, distortvec)
        # 计算每个点的误差
        error = img_points_projected - imgp[i]
        # total_error=np.sqrt(error[:,0]**2+error[:,1]**2)
        total_error.extend(error.flatten())  # 将误差展平成一维数组并添加到总误差中
    return np.array(total_error)
     
# def calculateError(K,RT,objp,imgp):
#     distance=[]
#     total_distance=0.00
#     for row1,row2 in zip(objp,imgp):
#       zeros=np.zeros((3,1))
#       K1=np.hstack((K,zeros))
#       row1=np.append(row1,1)
#        # 计算在相机坐标系中的`坐标
#       camera_coord = np.dot(RT, row1)
#         # 投影到图像平面
#       uvw = np.dot(K1, camera_coord)
#       uv = uvw[:2] / uvw[2]
#       distance.append(uv)
#       error = np.linalg.norm(uv - row2)
#       total_distance += error 
#     distance=np.array(distance)
#     return total_distance/len(objp)