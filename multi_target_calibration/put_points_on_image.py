#!/usr/bin/env python3

"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""

import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2

def points_on_image(undistorted_img, point_cloud, rotation_translation, intrinsic, veloyne=False):
    color_scale = 255 / 3
    p = np.matmul(intrinsic, rotation_translation)
    test_array = np.array(list(pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
    
    test_array = np.transpose(test_array)
    reflection = test_array[3, :].copy()
    test_array[3, :] = 1
    test_array = np.matmul(p, test_array)
    
    test_array = np.array([test_array[0, :] / test_array[2, :],
                           test_array[1, :] / test_array[2, :],
                           reflection]).T
    test_array = test_array.astype(int)
    
    (rows, cols, channels) = undistorted_img.shape
    for cor in test_array:
        px_c, px_r, color = cor
        
        # Ensure the color value is an integer and clamp it within a valid range (0 to 255)
        color = int(np.clip(color, 0, 255))
        
        if 0 <= px_c < cols and 0 <= px_r < rows:
            cv2.circle(undistorted_img, (px_c, px_r), 1, (140, 70, color), -1)
    
    cv2.imshow("Lidar points on image", undistorted_img)
    cv2.waitKey(3)

