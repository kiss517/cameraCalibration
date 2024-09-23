from matplotlib.pylab import show
import numpy as np
import cv2
import glob
from calculateErr import project_points
from cameraCalibration import cameraCalibrate
from showpicture import showpic

# 设置棋盘格规格
chessboard_size = (13, 8)
# 设置棋盘格内角点的真实世界坐标（假设棋盘格每个方格大小为1单位）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
# 准备存储所有图像的角点位置
objpoints = [] # 3D点
imgpoints = [] # 2D点
# 读取所有的棋盘格图像
images = glob.glob('picture/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # 如果找到角点，则添加到列表中
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners.reshape(-1, 2))
# 执行摄像机标定
mtx, dist, rvecs, tvecs = cameraCalibrate(objpoints, imgpoints)
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出标定结果
print("内参矩阵 (Camera matrix):\n", mtx)
print("畸变系数 (Distortion coefficients):\n", dist)
print("旋转向量",rvecs)
print("平移向量",tvecs)
# 重新投影误差
total_error = 0
for i,fname in enumerate(images):
    
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2=imgpoints2.reshape(-1,2)  
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
    project_promotion=project_points(mtx,objpoints[i],rvecs[i],tvecs[i],dist.reshape(-1))

    img = cv2.imread(fname)
    showpic(img,imgpoints[i],imgpoints2)
    
print("总体投影误差 (Total re-projection error): ", total_error / len(objpoints))


# params=[]
# params.extend(dist.reshape(-1))
# print(params)
# params.append(mtx[0,0])
# params.append(mtx[0,2])
# params.append(mtx[1,1])
# params.append(mtx[1,2])
# for i in range(len(objpoints)):
#  params.extend(rvecs[i].reshape(-1))
#  params.extend(tvecs[i].reshape(-1))
# params=np.array(params)

# # for i in range(len(objpoints)):
# #     imgpoints2 = project_points(mtx,objp[i],rvecs[i].reshape(-1),tvecs[i].reshape(-1),dist.reshape(-1))
# #     error = imgpoints[i]-imgpoints2
    
# #     tl.extend(error.flatten())
# error= calculate_reprojection_error(params,objpoints,imgpoints)
# total_error=np.linalg.norm(error)
# num=len(objpoints)*len(objpoints[0])
# print("优化后的重投影误差",np.sqrt(total_error*total_error/num))    


