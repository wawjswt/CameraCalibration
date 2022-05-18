# Editor:Wentian_Shen
# Coding in Python by PyCharm

import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("E:/img/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    cv2.namedWindow("img0", 0)
    cv2.resizeWindow("img0", 1080, 840)
    cv2.imshow('img0', img)
# 输出图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#转换为灰度图
    size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    print(fname[7:9] + '读取结果:' + str(ret))

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        # print(corners2)  调试用
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
# 调用函数绘制棋盘格
        cv2.namedWindow("img1", 0)
        cv2.resizeWindow("img1", 1080, 840)
        cv2.imshow('img1', img)
        cv2.waitKey(1000)

print('\n一共有' + str(len(img_points)) + "张照片\n")
print("正在计算中")
cv2.destroyAllWindows()

# 标定相机，计算参数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

np.savez("E:/abc.npz", mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
# 保存计算的参数结果，用npz格式保存元组

print("ret:", ret)
print("mtx:\n", mtx)        # 内参矩阵
print("\ndist:\n", dist)    # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("\nrvecs:\n", rvecs)  # 旋转向量  # 外参数
print("\ntvecs:\n", tvecs)  # 平移向量  # 外参数

print("*The End Of Calibration*")

