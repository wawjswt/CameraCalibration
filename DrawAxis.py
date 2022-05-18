# Editor:Wentian_Shen
# Coding in Python by PyCharm
# 绘制坐标轴和3D方块
import cv2
import numpy as np
import glob


# 构建函数drawaxis，用棋盘中的角点绘制 3D 坐标轴xyz。
def drawaxis(img, corners, imgpts):
    start = tuple(corners[0].ravel())
    start = [int(start[0]), int(start[1])]
    # 坐标原点

    end1 = tuple(imgpts[0].ravel())
    end1 = [int(end1[0]), int(end1[1])]
    # x轴

    end2 = tuple(imgpts[1].ravel())
    end2 = [int(end2[0]), int(end2[1])]
    # y轴

    end3 = tuple(imgpts[2].ravel())
    end3 = [int(end3[0]), int(end3[1])]

    # z轴
    img = cv2.line(img, start, end1, (235, 206, 135), 3)
    img = cv2.line(img, start, end2, (0, 255, 0), 3)
    img = cv2.line(img, start, end3, (0, 0, 255), 3)
    return img


def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # 用粉红色绘制底层
    img = cv2.drawContours(img, [imgpts[:4]], -1, (200, 198, 255), -3)

    # 用蓝色绘制垂直棱
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # 用红色绘制顶层棱
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# 从之前的标定结果中加载相机矩阵、畸变系数
with np.load('E:/abc.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# 坐标轴的3个点，X 轴是从(0,0,0) 到 (3,0,0) 绘制的，Y 轴是从(0,0,0) 到 (0,3,0)，Z 轴从(0,0,0) 到 (0,0,-3) 绘制。
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axis1 = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                    [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
# 绘制的是一个1X1X1的立方体

for fname in glob.glob('E:/img/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)

        ortho, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # 投影3D点到图像平面
        imgpts, jac = cv2.projectPoints(axis1, rvecs, tvecs, mtx, dist)
        cube_img = drawCube(img, corners2, imgpts)
        imgpts, jac1 = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = drawaxis(cube_img, corners2, imgpts)
        cv2.namedWindow("xyz", 0)
        cv2.resizeWindow("xyz", 1080, 860)
        cv2.imshow('xyz', img)
        cv2.waitKey(2000)

        cv2.imwrite('E:/axis/Wentian_Shen_' + fname[7:9] + '.jpg', img)

cv2.destroyAllWindows()
print("完成绘制")
