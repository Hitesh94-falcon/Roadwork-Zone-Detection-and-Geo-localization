#!/usr/bin/env python3

import numpy as np
import json

"""CHANGE THE PATH DIRECTORY IN THE MAIN()"""

#CLICKED_POINT_PATH = '/home/shyam/multi_targets_based_new/src/extrinsic_3d_to_camera/scripts/CLICKED.json'

#just load the data by opening the file as r . Check w3school examples
def load_extrinsic(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    extrinsic_data = data["top_center_lidar-to-center_camera-extrinsic"]["param"]["sensor_calib"]["data"]
    print(extrinsic_data)
    extrinsic_matrix = np.array(extrinsic_data)
    print("extrinsic_matrix")
    print(extrinsic_matrix)
    return extrinsic_matrix

def load_intrinsic(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    intrinsic_data = data["center_camera-intrinsic"]["param"]["cam_K"]["data"]
    intrinsic_matrix = np.array(intrinsic_data)
    print("intrinsic")
    print(intrinsic_matrix)
    return intrinsic_matrix

def load_points(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

        #find easy method to read and add the points
    
    points3d = [[point['x'], point['y'], point['z']] for point in data["clicked_points"]["3D_points"]]
    points2d = [[point['u'], point['v']] for point in data["clicked_points"]["2D_points"]]
    print (points3d)
    print(points2d)
    #read only 1st 3 points from 3D and 2D ... [:2] 
    return points3d[:3], points2d[:3]  ####Return only the first 3 points

#euclidean should read 2d points. best way to call a function in validation function to which all required data is sent
#this code will perform the validation and calculate error for all the points

def calculate_euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def validation(points3d, intrinsic, R, t, actual_points2d):
    #run loop
    for i in range(len(points3d)):  
        extrinsics = np.dot(R, np.transpose(points3d[i])) + t
        calc_points2d = np.dot(intrinsic, extrinsics)
        
        calc_points2d[0] /= calc_points2d[2]
        calc_points2d[1] /= calc_points2d[2]

        print("The actual point is: %s and the calculated point is : %s" %(actual_points2d[i], calc_points2d[:2]))
        
        error = calculate_euclidean_distance(calc_points2d[:2], actual_points2d[i])
        print("Error corresponding to the point %d: %s" % (i + 1, error))

        #write it to another json file or a normal text file
        
if __name__ == '__main__':
    #load the intrinsic, extrinsic, then R and T
    #remember to call the functions to first load the points from json and then send it to vlidate funstion

    intrinsic = load_intrinsic('/home/hitesh/Documents/Project/multi_target_calibration/cam1/intrinsic_cam1.json')
    extrinsic_matrix = load_extrinsic('/home/hitesh/Documents/Project/extrinsic_cam1__TEST.json')  
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    points3d, actual_points2d = load_points('/home/hitesh/Documents/Project/clicked points/clikedpoints camera1 to lidar.json') 
    validation(points3d, intrinsic, R, t, actual_points2d)
