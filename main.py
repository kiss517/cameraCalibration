import os
import tkinter as tk
from tkinter import filedialog, messagebox
from turtle import rt
import cv2
import numpy as np
from cameraCalibration import cameraCalibrate
from findRT import calculateRT
from EXAM.testB import examB
from EXAM.testCamera import examC
from EXAM.testK import examK
from EXAM.testRT import examRT
from EXAM.testV import examV
from calculateErr import calculate_reprojection_error, project_points
from cornerDetection import cornerDetection
from findB import findM_B
from findH import findHomography
from findK import findM_K1, findM_K2, findM_K3
from findOneV import  calculateOneV
from EXAM.test import examination
from optimism import optimized, optimizedResult
from showpicture import showpic
# 设置棋盘格大小和每个方格的实际尺寸
chessboard_size = (11, 8)

# 准备棋盘格图像的世界坐标系点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float64)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*23.35
img_paths = []
# 创建主窗口
root = tk.Tk()
root.title("选择图片")
root.geometry("800x600")
# 存储选中图片路径的列表

def select_all_images():

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png']
    for root, dirs, files in os.walk('picture'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_paths.append(os.path.join(root, file))
    update_selected_images()
    return 

# 处理图片选择
def select_images():
    try:
        # 确保所有常见的图片文件类型都被包括在内
        file_paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("JPEG Files", "*.jpg"), 
                       ("PNG Files", "*.png"), 
                       ("GIF Files", "*.gif"), 
                       ("Bitmap Files", "*.bmp"),
                       ("All Files", "*.*")]
        )
        if file_paths:
            img_paths.extend(file_paths)
            update_selected_images()
        else:
            messagebox.showinfo("提示", "没有选中的图片文件")
    except Exception as e:
        messagebox.showerror("错误", f"选择图片时出错: {str(e)}")

# 更新显示选中的图片路径
def update_selected_images():
    selected_images_text.delete(1.0, tk.END)
    for path in img_paths:
        selected_images_text.insert(tk.END, path + "\n")
 
# 使用OpenCV读取和显示图片
def display_images():
    obj_point=[]
    img_point=[]
    params=[0,0,0,0,0]
    vectorx=[]
    for path in img_paths:
          # 在函数内声明 vectorx 是全局变量
        image = cv2.imread(path)
        if image is not None:
             imgp=cornerDetection(image,11,8)
             obj_point.append(objp)
             img_point.append(imgp)
             H=findHomography(objp,imgp)
            #  examination(H,objp)
             V=calculateOneV(H)
             vectorx.append(V[0]) 
             vectorx.append(V[1])  
        else:
            messagebox.showerror("错误", f"无法读取图片: {path}")
   
    vectorx=np.array(vectorx)
    B=findM_B(vectorx)
    # examV(vectorx,B)
    # examB(H,B)
    print("K矩阵")
    K=findM_K3(B)
    print (K)
    # examK(K,B)
    params.append(K[0,0])
    params.append(K[0,2])
    params.append(K[1,1])
    params.append(K[1,2])
    i=1  
    for path in img_paths:
      image = cv2.imread(path)
      if image is not None:
        imgp=cornerDetection(image,11,8)
        H=findHomography(objp,imgp)
        RT,s=calculateRT(H,K)
        print(f'第{i}张图片的外参矩阵是')
        print(RT)
        # examRT(K,RT,H,s)
        # examC(RT,objp,K,s,H)
        R = RT[:3, :3]  # 3x3 旋转矩阵
        tvec = RT[:3, 3]  # 3x1 平移向量
        # 将旋转矩阵 R 转换为旋转向量 rvec
        rvec, _ = cv2.Rodrigues(R)
        rvec=rvec.ravel()
        params.extend(rvec)
        params.extend(tvec)
        i+=1
    
    # total_error_prime=np.linalg.norm(calculate_reprojection_error(params,obj_point,img_point))
    # print("total_error_prime",total_error_prime)
    # num=len(obj_point)*11*8
    # print("优化前的重投影误差",np.sqrt(total_error_prime*total_error_prime/num))
    optimized_params= optimized(params,obj_point,img_point)
    k1, k2, k3,p1, p2, fx, u0, fy, v0 = optimized_params[:9]
    camera_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    dist = np.array([k1, k2, k3,p1, p2])
    
    error= optimizedResult(optimized_params,obj_point,img_point)
    total_error=np.linalg.norm(error)
    num=len(obj_point)*len(obj_point[0])
    print("优化后的重投影误差",np.sqrt(total_error*total_error/num))
    # total_error=0.00
    # error=[]
    # for i in range(len(obj_point)):
    #      imgpoints2, _ = cv2.projectPoints(obj_point[i],optimized_params[9 + i * 6: 12 + i * 6], optimized_params[12 + i * 6: 15 + i * 6], 
    #                                        camera_matrix, dist)

    #      error_single= imgp[i]-imgpoints2
    #      total_error += error
    # print(total_error/ len(obj_point))

#优化前 
def beforeOptimize():
    obj_point=[]
    img_point=[]
    params=[0,0,0,0,0]
    vectorx=[]
    for path in img_paths:
          # 在函数内声明 vectorx 是全局变量
        image = cv2.imread(path)
        if image is not None:
             imgp=cornerDetection(image,11,8)
             obj_point.append(objp)
             img_point.append(imgp)
             H=findHomography(objp,imgp)
            #  examination(H,objp)
             V=calculateOneV(H)
             vectorx.append(V[0]) 
             vectorx.append(V[1])  
        else:
            messagebox.showerror("错误", f"无法读取图片: {path}")
   
    vectorx=np.array(vectorx)
    B=findM_B(vectorx)
    K=findM_K2(B)
    params.append(K[0,0])
    params.append(K[0,2])
    params.append(K[1,1])
    params.append(K[1,2])
    i=1  
    for path in img_paths:
      image = cv2.imread(path)
      if image is not None:
        imgp=cornerDetection(image,11,8)
        H=findHomography(objp,imgp)
        RT,s=calculateRT(H,K)
        R = RT[:3, :3]  # 3x3 旋转矩阵
        tvec = RT[:3, 3]  # 3x1 平移向量
        # 将旋转矩阵 R 转换为旋转向量 rvec
        rvec, _ = cv2.Rodrigues(R)
        rvec=rvec.ravel()
        params.extend(rvec)
        params.extend(tvec)
        project_p=project_points(K,objp,rvec,tvec,params[:5])
        showpic(image,imgp,project_p)
        i+=1
    
#优化后 
def afterOptimize():
    obj_point=[]
    img_point=[]
    params=[0,0,0,0,0]
    vectorx=[]
    for path in img_paths:
          # 在函数内声明 vectorx 是全局变量
        image = cv2.imread(path)
        if image is not None:
             imgp=cornerDetection(image,11,8)
             obj_point.append(objp)
             img_point.append(imgp)
    #          H=findHomography(objp,imgp)
    #         #  examination(H,objp)
    #          V=calculateOneV(H)
    #          vectorx.append(V[0]) 
    #          vectorx.append(V[1])  
    #     else:
    #         messagebox.showerror("错误", f"无法读取图片: {path}")
   
    # vectorx=np.array(vectorx)
    # B=findM_B(vectorx)
    # K=findM_K3(B)
    # params.append(K[0,0])
    # params.append(K[0,2])
    # params.append(K[1,1])
    # params.append(K[1,2])
    # i=1  
    # for path in img_paths:
    #   image = cv2.imread(path)
    #   if image is not None:
    #     imgp=cornerDetection(image,11,8)
    #     H=findHomography(objp,imgp)
    #     RT,s=calculateRT(H,K)
    #     R = RT[:3, :3]  # 3x3 旋转矩阵
    #     tvec = RT[:3, 3]  # 3x1 平移向量
    #     # 将旋转矩阵 R 转换为旋转向量 rvec
    #     rvec, _ = cv2.Rodrigues(R)
    #     rvec=rvec.ravel()
    #     params.extend(rvec)
    #     params.extend(tvec)
    #     i+=1
    # optimized_params= optimized(params,obj_point,img_point)
    mtx, dist, rvecs, tvecs = cameraCalibrate(obj_point, img_point)
    for i, path in enumerate(img_paths):
        image = cv2.imread(path)
    #   if image is not None:
    #     imgp=cornerDetection(image,11,8)
    #     k1, k2, k3,p1, p2, fx, u0, fy, v0 = optimized_params[:9]
    #     camera_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    #     rvec = optimized_params[9 + i * 6: 12 + i * 6]
    #     tvec = optimized_params[12 + i * 6: 15 + i * 6]
        
        project_promotion=project_points(mtx,obj_point[i],rvecs[i],tvecs[i],dist.reshape(-1))
        # print(project_promotion)
        showpic(image,img_point[i],project_promotion)
    
#清空选择的图片
def clear_list():
    img_paths.clear()
    update_selected_images()

# 创建选择按钮
select_button = tk.Button(root, text="选择图片", command=select_images)
select_button.pack(pady=20)
# 创建全部选择图片按钮
display_button = tk.Button(root, text="全选图片", command=select_all_images)
display_button.pack(pady=20)

display_button = tk.Button(root, text="清空列表", command=clear_list)
display_button.pack(pady=20)

# 显示选中图片路径的文本框
selected_images_text = tk.Text(root, wrap=tk.WORD, height=10, width=70)
selected_images_text.pack(pady=20)

# 创建一个Frame来放置三个按钮
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# 创建三个按钮
button1 = tk.Button(button_frame, text="显示图片", command=display_images)
button2 = tk.Button(button_frame, text="优化前的重投影", command=beforeOptimize)
button3 = tk.Button(button_frame, text="优化后的重投影", command=afterOptimize)

# 使用grid布局管理器将按钮平行放置在最后一排
button1.grid(row=0, column=0, padx=10)
button2.grid(row=0, column=1, padx=10)
button3.grid(row=0, column=2, padx=10)



# 运行主循环
root.mainloop()
