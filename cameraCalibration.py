import cv2
from calculateErr import  project_points
from findB import findM_B
from findH import findHomography
from findK import findM_K1, findM_K2, findM_K3
from findOneV import  calculateOneV
import numpy as np
from findRT import calculateRT
from optimism import optimized, optimizedResult
def cameraCalibrate(obj_point,img_point):
# 设置棋盘格大小和每个方格的实际尺寸
#  objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float64)
#  objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*23.35
 params=[0,0,0,0,0]
 vectorx=[]
 for objp,imgp in zip(obj_point,img_point):
             H=findHomography(objp,imgp)
             V=calculateOneV(H)
             vectorx.append(V[0]) 
             vectorx.append(V[1])  
 vectorx=np.array(vectorx)
 B=findM_B(vectorx)
 print("K矩阵")
 K=findM_K3(B)
 print (K)
 params.append(K[0,0])
 params.append(K[0,2])
 params.append(K[1,1])
 params.append(K[1,2])
 for i, (objp, imgp) in enumerate(zip(obj_point, img_point)):
        H=findHomography(objp,imgp)
        RT,s=calculateRT(H,K)
        print(f'第{i}张图片的外参矩阵是')
        print(RT)
        R = RT[:3, :3]  # 3x3 旋转矩阵
        tvec = RT[:3, 3]  # 3x1 平移向量
        # 将旋转矩阵 R 转换为旋转向量 rvec
        rvec, _ = cv2.Rodrigues(R)
        rvec=rvec.ravel()
        params.extend(rvec)
        params.extend(tvec)    
    # total_error_prime=np.linalg.norm(calculate_reprojection_error(params,obj_point,img_point))
    # print("total_error_prime",total_error_prime)
    # num=len(obj_point)*13*8
    # print("优化前的重投影误差",np.sqrt(total_error_prime*total_error_prime/num))
 optimized_params= optimized(params,obj_point,img_point)
 k1, k2, k3,p1, p2, fx, u0, fy, v0 = optimized_params[:9]
 camera_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
 dist = np.array([k1, k2, k3,p1, p2]).reshape(1,5)
 rvecs_list=[]
 tvecs_list=[]
 for i in range(len(obj_point)):
    rvec = optimized_params[9 + i * 6: 12 + i * 6]
    tvec = optimized_params[12 + i * 6: 15 + i * 6]
    rvecs_list.append(np.array(rvec, dtype=np.float64).reshape(-1,1))
    tvecs_list.append(np.array(tvec, dtype=np.float64).reshape(-1,1))
 rvecs = tuple(rvecs_list)
 tvecs=tuple(tvecs_list)
 return camera_matrix,dist,rvecs,tvecs