import numpy as np
import cv2

from showpicture import showpic
def cornerDetection(image,sizeA,sizeB):
 # 设置棋盘格大小和每个方格的实际尺寸
 chessboard_size = (sizeA, sizeB)
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 # 查找棋盘格角点
 ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
 if ret:
        # 提高角点的精确度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 将角点坐标存储到一个NumPy数组中，并进行排序
        corners_array = corners2.reshape(-1, 2)
        sorted_corners=np.array(corners_array)
 else:
    print('棋盘格角点未找到')
   
 return sorted_corners
 