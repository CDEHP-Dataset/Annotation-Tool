import json
import os
import numpy as np
import pyrealsense2 as rs
import cv2

class CoordinateConverter:
    def __init__(self):
        with open(r"C:\Users\Xavier\Desktop\biaoding\intrinsics.json") as f:
            intrinsics = json.load(f)
        self.data = None
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = intrinsics["depth"]["width"]
        self.intrinsics.height = intrinsics["depth"]["height"]
        self.intrinsics.fx = intrinsics["depth"]["fx"]
        self.intrinsics.fy = intrinsics["depth"]["fy"]
        self.intrinsics.ppx = intrinsics["depth"]["ppx"]
        self.intrinsics.ppy = intrinsics["depth"]["ppy"]
        self.intrinsics.coeffs = intrinsics["depth"]["coeffs"]
        self.depth_scale = intrinsics["depth_scale"]

        self.cam_matrix_left = np.load(r"C:\Users\Xavier\Desktop\biaoding\annt\m_l.npy")
        self.cam_matrix_right = np.load(r"C:\Users\Xavier\Desktop\biaoding\annt\m_r.npy")
        self.R = np.load(r"C:\Users\Xavier\Desktop\biaoding\annt\R.npy")
        self.T = np.load(r"C:\Users\Xavier\Desktop\biaoding\annt\T.npy")
    def get_m(self):
        r_t = np.hstack([self.R, self.T])
        temp = np.zeros(len(r_t[0]))
        temp[-1] = 1
        r_t = np.vstack([r_t, temp])
        m_l = np.hstack([self.cam_matrix_left, np.array([[0]] * len(self.cam_matrix_left))])
        m_r = np.dot(np.hstack([self.cam_matrix_right, np.array([[0]] * len(self.cam_matrix_right))]), r_t)
        return m_l,m_r
    def convert(self, x: int, y: int,data):
        distance = data[y][x]
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], distance)
        camera_coordinate = np.array(camera_coordinate).reshape(3,1)
        m_l,m_r = self.get_m()
        s_r = np.dot(m_r[-1], np.vstack([camera_coordinate, [1]]))
        r_pi = np.dot(m_r, np.vstack([camera_coordinate, [1]])) / s_r
        return r_pi[0],r_pi[1]

if __name__ == '__main__':
    CV = CoordinateConverter()
    data_depth = np.load(r"C:\Users\Xavier\Desktop\0520\depth_raw\depth_07.npy")
    test_x_y = np.array([[77, 426], [227, 339], [405, 225], [272, 136], [656, 267], [590, 90], [467, 272]])
    point_test_3d = []
    for i in range(len(test_x_y)):
        x,y = CV.convert(test_x_y[i][0], test_x_y[i][1],data_depth)
        point_test_3d.append([x,y])
    print(point_test_3d)
    image_l = cv2.imread(r"C:\Users\Xavier\Desktop\0520\color\d435i_07.png")
    image_r = cv2.imread(r"C:\Users\Xavier\Desktop\0520\image_event_binary\celex5_07.png")
    image_r[392, :, :] = (255, 0, 0)
    image_r[:, 638, :] = (255, 0, 0)
    for i in range(len(point_test_3d)):
        image_l[test_x_y[i][1] - 2:test_x_y[i][1] + 2, test_x_y[i][0] - 2:test_x_y[i][0] + 2, :] = (0, 0, 255)
        image_r[int(point_test_3d[i][1][0]) - 2:int(point_test_3d[i][1][0]) + 2, int(point_test_3d[i][0][0]) - 2:int(point_test_3d[i][0][0]) + 2, :] = (0, 0, 255)
    cv2.imshow("img_l", image_l)
    cv2.waitKey(0)
    cv2.imshow("img_r", image_r)
    cv2.waitKey(0)