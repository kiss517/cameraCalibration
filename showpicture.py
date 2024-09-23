import cv2
def showpic(image,sorted_corners,project_points):
    for i, corner in enumerate(sorted_corners):
        cv2.putText(image, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (int(corner[0]), int(corner[1])), 10, (0, 255, 0), -1)

    for i, corner in enumerate(project_points):
        cv2.putText(image, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(image, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
      #  调整图像大小
    fixed_size = (1000, 800)  # 例如，将图像调整为800x600
    resized_image = cv2.resize(image, fixed_size)
     # 创建一个窗口并设置窗口属性为可调整大小
    window_name = 'Chessboard corners'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

     # 显示调整后的图像
    cv2.imshow(window_name,resized_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return