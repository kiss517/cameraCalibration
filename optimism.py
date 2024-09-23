from scipy.optimize import least_squares
import numpy as np
import cv2
from calculateErr import calculate_reprojection_error, project_points

def optimized(params, objp, imgp):
    # 使用最小二乘法优化参数
    result = least_squares(calculate_reprojection_error, params, args=(objp, imgp), method='lm')
    optimized_params = result.x
    return optimized_params
def  optimizedResult(optimized_params,objp,imgp):

   # 提取优化后的相机内参和畸变系数
    k1, k2, k3,p1, p2, fx, u0, fy, v0 = optimized_params[:9]
    camera_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, k3,p1, p2])

    # 打印优化后的相机内参和畸变系数
    print("camera_matrix:\n", camera_matrix)
    print("Optimized Distortion Coefficients:\n", dist_coeffs)

    for i in range(len(objp)):
        rvec = optimized_params[9 + i * 6: 12 + i * 6]
        tvec = optimized_params[12 + i * 6: 15 + i * 6]
        project_points(camera_matrix, objp[i], rvec, tvec, dist_coeffs)
    error=calculate_reprojection_error(optimized_params,objp,imgp)
    return error