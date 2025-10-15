import os
import cv2
import numpy as np
import open3d as o3d
import math
import json
from PIL import Image
from scipy.spatial.transform import Rotation
import copy
import imageio

num_obj_positions = 5 # cross patern

def generate_camera_targets(camera_targets, sigma_targets, random = False):
    targets = []
    
    for i in range(len(camera_targets)):
        target = []
        for j in range(3):
            target.append(camera_targets[i][j] + generate_random_offset(0,sigma_targets,random))
        targets.append(target)
    
    return targets
        
def generate_camera_origins(configuration, d1, d2, heights, sigma_height, random_height = False):
    origins = []
    height = heights[0] # temp, must be fixed
    if(configuration == "circle"):
        for i in range(len(heights)):
           origins.append([0,generate_random_offset(0, sigma_height, random_height) + height, d1])
    else:
        assert(len(heights) == 8)
        origins = [[0,generate_random_offset(0, sigma_height, random_height) + height, d1], [d2,generate_random_offset(0, sigma_height, random_height) + height,d1], [d2,generate_random_offset(0, sigma_height, random_height) + height,0], [d2,generate_random_offset(0, sigma_height, random_height) + height,-d1], [0,generate_random_offset(0, sigma_height, random_height) + height,-d1], [-d2,generate_random_offset(0, sigma_height, random_height) + height,-d1], [-d2,generate_random_offset(0, sigma_height, random_height) + height,0], [-d2,generate_random_offset(0, sigma_height, random_height) + height,d1]]
    return origins

def generate_obj_positions_and_angles(configuration, d1, d2):
    if(configuration == "rectangle"):
        positions_flat = [[0,0,0], [0,0,d1], [0,0,-d1], [d2,0,0], [-d2,0,0]]
        positions_perp = [[0,0.45,0], [0,0.45,d1], [0,0.45,-d1], [d2,0.45,0], [-d2,0.45,0]]
    else:
        positions_flat = [[0,0,0], [0,0,d1], [0,0,-d1], [d1,0,0], [-d1,0,0]]
        positions_perp = [[0,0.45,0], [0,0.45,d1], [0,0.45,-d1], [d1,0.45,0], [-d1,0.45,0]]
    
    positions = positions_flat + positions_perp
    
    obj_angles_flat = [[0,j,0] for j in [x for x in range(0,360,30)]]
    obj_angles_perp = [[90,j,0] for j in [x for x in range(0,360,30)]]
    obj_angles_perp2 = [[90,0,j] for j in [x for x in range(-50,50,45)]]
    
    angles = [obj_angles_flat] * len(positions_flat) + [obj_angles_perp] * len(positions_perp)
    
    return positions, angles
            
def generate_random_offset(mu, sigma, random_on):
    if(not random_on):
        num = 0
    else:
        num = np.random.normal(mu, sigma)
            
    return num

def save_pgm(aov_image, sensor, filename):
    aov_np = np.array(aov_image)
    
    pos_world = aov_np[:, :, :3]  # (x, y, z)

    # Transform positions into camera (sensor) local space
    cam_to_world = sensor.world_transform().matrix
    world_to_cam = np.linalg.inv(cam_to_world)

    # Convert to homogeneous coordinates
    H, W, _ = pos_world.shape
    # Convert to homogeneous coordinates (H*W, 4)
    pos_h = np.ones((H * W, 4))
    pos_h[:, :3] = pos_world.reshape(-1, 3)

    # Apply world-to-camera transform
    pos_cam = (world_to_cam @ pos_h.T).T  # Shape: (H*W, 4)

    # Reshape back to image
    pos_cam = pos_cam.reshape(H, W, 4)

    # Extract z-coordinate in camera space (depth)
    depth_np = pos_cam[:, :, 2]
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    depth_16bit = (depth_np * 1000).astype(np.uint16)
    #Image.fromarray(depth_16bit, mode='I;16').save(filename)
    
    # Use depth aov from mitsuba instead of position
    depth_np = aov_np[:, :, 9]
    depth_np_mm = depth_np * 1000
    depth_16bit = (depth_np_mm).astype(np.uint16)
    Image.fromarray(depth_16bit, mode='I;16').save(filename)

    #Temp until pgm is correct: save depth as tiff
    #Image.fromarray(depth_np).save(filename + "_test.tif")

def save_ply(aov_image, downsample, voxel_size, filename, save_pcd = False, gt_extrinsics = None, base=None, add_noise=False, noise_std=0):
    width=1320
    height=720
    
    aov_np = np.array(aov_image)
    depth_np = aov_np[:, :, 0:3] * [1,-1,1] 
    normal_np = aov_np[:, :, 3:6]
    rgb_np = aov_np[:, :, 6:9]
    rgb_np = np.clip(rgb_np, 0, 1)
    
    
    positions = np.array(depth_np).reshape(height, width, 3)  * np.array([1, -1, 1])
    points = positions.reshape(-1, 3)
    
    if(add_noise):
        noise = np.random.normal(0, noise_std, points.shape)
        points = points + noise
    
    colors = np.array(rgb_np).reshape(height, width, 3)
    colors = colors.reshape(-1, 3)
    
    normals = np.array(normal_np).reshape(height, width, 3)
    normals = normals.reshape(-1, 3)
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    #For gicp save plys in camera world coordinates, so that they are not already aligned
    if gt_extrinsics is not None:
        if base is not None:
            pcd.transform(base)
        pcd.transform(np.linalg.inv(np.array(gt_extrinsics)))

    # clean pointcloud to avoid artifacts
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    if(downsample):
        pcd = pcd.voxel_down_sample(voxel_size)
    
    if(save_pcd):
        o3d.io.write_point_cloud(filename + ".pcd", pcd)
    
    o3d.io.write_point_cloud(filename + ".ply", pcd)
    
def add_noise_to_extr_dir(extr_dir, sigma_rot, sigma_trans, rebase = True, add_noise=True):
    noisy_extr_dir = {}
    cam_keys = list(extr_dir.keys())
    
    T_mi_cam_to_cv_cam = np.eye(4)
    #T_mi_cam_to_cv_cam[0,0] = -1
    #T_mi_cam_to_cv_cam[1,1] = -1
    T_cv_cam_to_mi_cam = np.linalg.inv(T_mi_cam_to_cv_cam)
    
    T_cv_w_to_mi_w = np.array([
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                            ])
    T_mi_w_to_cv_w = np.linalg.inv(T_cv_w_to_mi_w)
    
    for key in cam_keys:
        # Convert to opencv coordinate system assuming gicp worl = mitsuba world
        T_mi_cam_to_mi_w = np.array(extr_dir[key])
        T_cv_cam_to_cv_world = T_mi_w_to_cv_w @ T_mi_cam_to_mi_w @ T_cv_cam_to_mi_cam
        
        if(add_noise == False):
            noisy_extr_dir[key] = T_mi_cam_to_mi_w.tolist()
            continue
        
        # Add random rotation and translation to matrix before saving to json...
        R = np.array(T_mi_cam_to_mi_w)[:3, :3]
        t = np.array(T_mi_cam_to_mi_w)[:3, 3]
        
        # Rotation noise
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle_rad = np.deg2rad(np.random.normal(0.0, sigma_rot))
        rot_noise = Rotation.from_rotvec(angle_rad * axis).as_matrix()
        R_noisy = rot_noise @ R

        # Translation noise
        t_noise = np.random.normal(loc=0.0, scale=sigma_trans, size=3)
        t_noisy = t + t_noise

        # Rebuild transformation matrix
        T_noisy = np.eye(4)
        T_noisy[:3, :3] = R_noisy
        T_noisy[:3, 3] = t_noisy
        
        noisy_extr_dir[key] = T_noisy.tolist() 
    
    base_inv = None
    if(rebase):
        base_inv = np.linalg.inv(noisy_extr_dir[cam_keys[0]])

        for k in cam_keys:
            noisy_extr_dir[k] = np.round((base_inv @ noisy_extr_dir[k]), decimals=10).tolist()

    return noisy_extr_dir, base_inv
    
# For structureNet
def save_device_repository(serials, output_folder):
    cx, cy, fx, fy = compute_intrinsics(90, 1320, 720)
    device_repository = []
    
    key1 = "1320x720"
    
    for serial in serials:
        device_rep = { 
                      "Device": serial, "Serial": serial, 
                      "Color Intrinsics": [{key1: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 ]}], 
                      "Depth Intrinsics": [{key1: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 ]}], 
                      "Depth Color Rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      "Depth Color Translation": [0,0,0],
                      "Depth Radial Distortion Coeffs": [0, 0, 0, 0, 0, 0],
                      "Color Radial Distortion Coeffs": [0, 0, 0, 0, 0, 0],
                      "Depth Tangential Distortion Coeffs": [0, 0],
                      "Color Tangential Distortion Coeffs": [0, 0]
                    }
        device_repository.append(device_rep)
    
    config_json = os.path.join(output_folder, "device_repository.json")
    with open(config_json, 'w') as f:
        json.dump(device_repository, f, indent=4)
    
def compute_intrinsics(fov, width, height):
    cx = width/2        #to comply with opencv center of image
    cy = height/2
    
    fov_x_rad = math.radians(fov)
    fov_y_rad = 2 * math.atan((height / width) * math.tan(fov_x_rad / 2))
    
    fx = width/(2*math.tan(fov_x_rad/2))
    fy = height/(2*math.tan(fov_y_rad/2))
    
    print(cx, cy, fx, fy)
    exit()
    
    return cx, cy, fx, fy

def distort_image_opencv(img, distCoeff):
    h, w = img.shape[:2]
    new_size = (w, h)
    
    cx, cy, fx, fy = compute_intrinsics(90, w, h)
    K = np.array([[fx,  0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Negate the distortion coefficients
    k1, k2, p1, p2, k3 = distCoeff
    neg_dist = np.array([-k1, -k2, -p1, -p2, -k3], dtype=np.float32)

    # Compute the 2× map for distortion (the “inverse” of undistort)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, neg_dist, None, K, new_size, cv2.CV_32FC1
    )

    # Remap
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def overwrite_as_Y(num_frames, camera_serials, dataset_folder, ):
    for idx in range(num_frames):
        for sensor_idx in range(len(camera_serials)):
            cam_folder = os.path.join(dataset_folder, "cam_"+ camera_serials[sensor_idx] + "/")
            if not os.path.exists(cam_folder):
                print("ERROR: Camera folder missing")
                exit()
                
            image = cv2.imread(cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png")
            ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            Y_component = ycrcb_image[:, :, 0]
            cv2.imwrite(cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png", Y_component)
            