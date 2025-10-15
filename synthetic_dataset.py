import mitsuba
import json
import shutil
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import math
import sys
import argparse
import object_handler
import helper_functions
from config_handler import Config
import time
from compute_rae import load_points_from_ply_folder, compute_icp_pointwise_RAE

mitsuba.set_variant("cuda_ad_rgb")

from mitsuba import ScalarTransform4f as T

dataset_info = {}
gt_extrinsics_all = {}

def create_combined_rotation(obj_rot, order_of_rotations=[3,2,1]):
    x_rotation_degrees = obj_rot[0]
    y_rotation_degrees = obj_rot[1]
    z_rotation_degrees = obj_rot[2]

    rotation_x = T.rotate(axis=[1, 0, 0], angle=x_rotation_degrees)
    rotation_y = T.rotate(axis=[0, 1, 0], angle=y_rotation_degrees)
    rotation_z = T.rotate(axis=[0, 0, 1], angle=z_rotation_degrees)

    if(order_of_rotations == [1,2,3]):
        return rotation_x @ rotation_y @ rotation_z
    elif(order_of_rotations == [1,3,2]):
        return rotation_x @ rotation_z @ rotation_y
    elif(order_of_rotations == [2,1,3]):
        return rotation_y @ rotation_x @ rotation_z
    elif(order_of_rotations == [2,3,1]):
        return rotation_y @ rotation_z @ rotation_x
    elif(order_of_rotations == [3,1,2]):
        return rotation_z @ rotation_x @ rotation_y
    elif(order_of_rotations == [3,2,1]):
        return rotation_z @ rotation_y @ rotation_x
    
def load_scene_debug(obj_rot, obj_pos=[0,0,0], order_of_rotations=[3,2,1]):
    my_scene = {
        'type': 'scene',
        'id': 'my_scene',
        'integrator': {
            'type': 'path'
        }
    }

    combined_rotation = create_combined_rotation(obj_rot, order_of_rotations)

    object_dict = {
        'type': "obj",
        'filename': board_obj,
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "bitmap",
                "filename": board_texture,
                "wrap_mode": "clamp"
            }
        },
        'to_world':  T.translate(obj_pos) @ combined_rotation
    }
    my_scene["board"] = object_dict
    
    room_dict = object_handler.get_room_objects(config.room)
    my_scene.update(room_dict)
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,0,0]
            }
        },
        'to_world':  T.translate([0,0.06,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["center"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [1,0,0]
            }
        },
        'to_world':  T.translate([0.1,0.06,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["x1"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [1,0,0]
            }
        },
        'to_world':  T.translate([0.2,0.06,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["x2"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [1,0,0]
            }
        },
        'to_world':  T.translate([0.3,0.06,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["x3"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [1,0,0]
            }
        },
        'to_world':  T.translate([0.4,0.06,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["x4"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,1,0]
            }
        },
        'to_world':  T.translate([0,0.16,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["y1"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,1,0]
            }
        },
        'to_world':  T.translate([0,0.26,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["y2"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,1,0]
            }
        },
        'to_world':  T.translate([0,0.36,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["y3"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,1,0]
            }
        },
        'to_world':  T.translate([0,0.46,0]).scale([0.02,0.02,0.02]) 
    }
    my_scene["y4"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,0,1]
            }
        },
        'to_world':  T.translate([0,0.06,0.1]).scale([0.02,0.02,0.02]) 
    }
    my_scene["z1"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,0,1]
            }
        },
        'to_world':  T.translate([0,0.06,0.2]).scale([0.02,0.02,0.02]) 
    }
    my_scene["z2"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,0,1]
            }
        },
        'to_world':  T.translate([0,0.06,0.3]).scale([0.02,0.02,0.02]) 
    }
    my_scene["z3"] = object_dict
    
    object_dict = {
        'type': "sphere",
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "rgb",
                "value": [0,0,1]
            }
        },
        'to_world':  T.translate([0,0.06,0.4]).scale([0.02,0.02,0.02]) 
    }
    my_scene["z4"] = object_dict
    
    light = {
        'type': 'envmap',
        'filename': config.envmap,
        'to_world': T.translate([0,0,0]).rotate(config.envmap_rot_axis, config.envmap_rot_degrees),
        'scale': config.envmap_scale
    }

    my_scene['light'] = light

    return mitsuba.load_dict(my_scene)

def load_scene(obj_rot, obj_pos=[0,0,0], order_of_rotations=[3,2,1], mode = ""):
    my_scene = {
        'type': 'scene',
        'id': 'my_scene',
        'integrator': {
            'type': 'path'
        }
    }
    combined_rotation = create_combined_rotation(obj_rot, order_of_rotations)

    if(mode != "empty" and mode != "boxes"):
        object_dict = {
            'type': "obj",
            'filename': board_obj,
            'bsdf': {
                "type": "diffuse",
                    "reflectance": {
                    "type": "bitmap",
                    "filename": board_texture,
                    "wrap_mode": "clamp"
                }
            },
            'to_world':  T.translate(obj_pos) @ combined_rotation
        }   
    
        my_scene["board"] = object_dict
    elif(mode == "boxes"):
        boxes_dict = object_handler.get_room_objects(config.boxes)
        my_scene.update(boxes_dict)
    
    room_dict = object_handler.get_room_objects(config.room)
    my_scene.update(room_dict)

    light = {
        'type': 'envmap',
        'filename': config.envmap,
        'to_world': T.translate([0,0,0]).rotate(config.envmap_rot_axis, config.envmap_rot_degrees),
        'scale': config.envmap_scale
    }

    my_scene['light'] = light

    return mitsuba.load_dict(my_scene)

def load_scene_depth(obj_rot, obj_pos=[0,0,0], order_of_rotations=[3,2,1], mode = ""):
    my_scene = {
        'type': 'scene',
        'id': 'my_scene_depth',
        'integrator': {
            'type': 'aov', 
            'aovs': 'dd.y:position, nn:sh_normal, albedo:albedo, dd.z:depth'
        },
    }
    combined_rotation = create_combined_rotation(obj_rot, order_of_rotations)

    if(mode != "empty" and mode != "boxes"):
        object_dict = {
            'type': "obj",
            'filename': board_obj,
            'bsdf': {
                "type": "diffuse",
                    "reflectance": {
                    "type": "bitmap",
                    "filename": board_texture,
                    "wrap_mode": "clamp"
                }
            },
            'to_world':  T.translate(obj_pos) @ combined_rotation
        }
    
        my_scene["board"] = object_dict
        
    if(mode == "boxes"):
        boxes_dict = object_handler.get_room_objects(config.boxes)
        my_scene.update(boxes_dict)
    
    room_dict = object_handler.get_room_objects(config.room)
    my_scene.update(room_dict)

    light = {
        'type': 'envmap',
        'filename': config.envmap,
        'to_world': T.translate([0,0,0]).rotate(config.envmap_rot_axis, config.envmap_rot_degrees),
        'scale': config.envmap_scale
    }

    my_scene['light'] = light

    return mitsuba.load_dict(my_scene)

def render_dataset_images(sensors, obj_positions, obj_angles, center_cam_idx, center_obj_pos, camera_serials, dataset_folder, output_path = ""):
    idx = 0
    
    for i in range(len(obj_positions)):
        for obj_rot in obj_angles[i]:
            obj_pos = obj_positions[i]
           
            order_of_rotations=[3,2,1]
            my_scene = load_scene(obj_rot, obj_pos, order_of_rotations)
            
            if(idx%20 == 0):
                print("Frame", idx, "rot=", obj_rot, "pos=", obj_pos)
            for sensor_idx in range(len(sensors)):
                cam_folder = os.path.join(dataset_folder, "cam_"+ camera_serials[sensor_idx] + "/")
                if not os.path.exists(cam_folder):
                    os.mkdir(cam_folder)
                    
                image = mitsuba.render(scene=my_scene, spp=config.spp, sensor=sensors[sensor_idx])
                
                if(config.add_noise):
                    noise = np.random.normal(loc=0.0, scale=config.noise_std, size=image.shape)
                    image = np.clip(image + noise, 0.0, 1.0) 
                
                img_filename = cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png"
                mitsuba.util.write_bitmap(img_filename, image)
                if(config.add_distortion):
                    img = None
                    while(img is None):
                        img = cv2.imread(img_filename)
                    img = helper_functions.distort_image_opencv(np.array(img), config.dist_coeffs)
                    cv2.imwrite(img_filename, img)
                    
                #----------------------Save ply---------------------
                if(config.save_ply):
                    my_scene_depth = load_scene_depth(obj_rot, obj_pos, order_of_rotations)
                    aov_image = mitsuba.render(scene=my_scene_depth, spp=132, sensor=sensors[sensor_idx])
                    
                    filename = cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + "_pointcloud.ply"
                    helper_functions.save_ply(aov_image, config.downsample, config.voxel_size, filename)
            
            idx += 1
    
    #Add a frame with the checkerboard horizontal 
    dataset_info["calibration"]["center_img_name"] = str(idx).zfill(3)

    my_scene = load_scene([0,180,0], center_obj_pos)
    my_scene_debug = load_scene_debug([0,180,0], center_obj_pos)
   
    for sensor_idx in range(len(sensors)):
        cam_folder = os.path.join(dataset_folder, "cam_"+ camera_serials[sensor_idx] + "/")
        if not os.path.exists(cam_folder):
            os.mkdir(cam_folder)            
        image = mitsuba.render(scene=my_scene, spp=config.spp, sensor=sensors[sensor_idx])
        
        img_filename = cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png"
        mitsuba.util.write_bitmap(img_filename, image)
        if(config.add_distortion):
            img = None
            while(img is None):
                img = cv2.imread(img_filename)
            img = helper_functions.distort_image_opencv(img, config.dist_coeffs)
            cv2.imwrite(img_filename, img)
                                                                                                
        if(sensor_idx == center_cam_idx):
            image = mitsuba.render(scene=my_scene_debug, spp=config.spp, sensor=sensors[sensor_idx])
            
            img_filename = output_path + "\\mitsuba_axes_orientation_" + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png"
            mitsuba.util.write_bitmap(img_filename, image)
            if(config.add_distortion):
                img = None
                while(img is None):
                    img = cv2.imread(img_filename)
                img = helper_functions.distort_image_opencv(img, config.dist_coeffs)
                cv2.imwrite(img_filename, img)
    
    
    idx += 1
    
    #Add a frame with no checkerboard in the scene 
    my_scene = load_scene([0,180,0], center_obj_pos, mode="empty")
    
    for sensor_idx in range(len(sensors)):
        cam_folder = os.path.join(dataset_folder, "cam_"+ camera_serials[sensor_idx] + "/")
        if not os.path.exists(cam_folder):
            os.mkdir(cam_folder)            
        image = mitsuba.render(scene=my_scene, spp=config.spp, sensor=sensors[sensor_idx])
        
        img_filename = cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + ".png"
        mitsuba.util.write_bitmap(img_filename, image)
        if(config.add_distortion):
            img = None
            while(img is None):
                img = cv2.imread(img_filename)
            img = helper_functions.distort_image_opencv(img, config.dist_coeffs)
            cv2.imwrite(img_filename, img)
                
        #----------------------Save ply---------------------
        if(config.save_ply):
            my_scene_depth = load_scene_depth([0,180,0], center_obj_pos, mode="empty")
            aov_image = mitsuba.render(scene=my_scene_depth, spp=132, sensor=sensors[sensor_idx])
            
            filename = cam_folder + camera_serials[sensor_idx] + "_" + str(idx).zfill(4) + "_pointcloud.ply"
            helper_functions.save_ply(aov_image, config.downsample, config.voxel_size, filename)
            

def rotate_camera_origin(origin, target, angle):
    theta = np.radians(angle)

    translated_camera_origin = np.array(origin) - np.array(target)
    rotation_axis = [0,1,0]

    R = np.empty([3, 3])
    if(rotation_axis == [0,0,1]):
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    elif(rotation_axis == [0,1,0]):
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
    else:
        R = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
        
    rotated_camera_origin = R @ translated_camera_origin

    new_camera_origin = rotated_camera_origin + target

    return new_camera_origin

def load_sensor(origin, target=[0,0,0], angle=0, fov=45, width=512, height=512, print_info = True):
    origin = rotate_camera_origin(origin, target, angle)
    
    off_x = helper_functions.generate_random_offset(0, config.sigma_origins, config.random_origins)
    off_z = helper_functions.generate_random_offset(0, config.sigma_origins, config.random_origins)
    
    off_y = 0
    if(config.shape == 'custom'):
        off_y = helper_functions.generate_random_offset(0, config.sigma_heights, config.random_heights)
    
    origin = [origin[0]+off_x, origin[1]+off_y, origin[2]+off_z]
   
    transform = T.look_at(origin=origin, target=target, up=[0,1,0])
    trans_matrix = np.array(transform.matrix)

    cx, cy, fx, fy = helper_functions.compute_intrinsics(fov, width, height)
    
    return cx, cy, fx, fy, trans_matrix, mitsuba.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': transform,
        'principal_point_offset_x': 0,  #normalized principal point, [0,0] -> center of image
        'principal_point_offset_y': 0,
        'sampler': {
            'type': 'multijitter',
            'sample_count': config.spp,
            'seed': 0
        },
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb'
        },
    })
  
def load_sensors(camera_serials, camera_angles, center_cam_idx, camera_origins, camera_targets):
    assert(len(camera_angles) == len(camera_serials))
    dataset_info["calibration"] = ({"n_cams": len(camera_angles), "cam_serials": camera_serials, "center_cam_idx": center_cam_idx})
    sensors = []
    
    ground_truth_extr_all = {}
    ground_truth_extr_pair = {}
    for sensor_idx in range(len(camera_serials)):
        print("camera" + str(sensor_idx))
        cx, cy, fx, fy, trans_matrix, sensor = load_sensor(origin=camera_origins[sensor_idx], target=camera_targets[sensor_idx], angle=camera_angles[sensor_idx], fov=90, width=1320, height=720, print_info=True)
        sensors.append(sensor)
        
        dataset_info["camera" + str(sensor_idx) + "_intrinsics"] = {"cx": cx, "cy": cy, "fx": fx, "fy":fy}
        dataset_info["camera_to_world_mi_coords_" + camera_serials[sensor_idx]] = {"rvec": cv2.Rodrigues(trans_matrix[:3,:3])[0][:,0].tolist(), "tvec": trans_matrix[:3,3].tolist()}

        ground_truth_extr_all[str(sensor_idx)] = trans_matrix.tolist()
        ground_truth_extr_pair[str(sensor_idx)] = trans_matrix.tolist()
        
        if(sensor_idx != 0):
            gt_extrinsics_pair, _ = helper_functions.add_noise_to_extr_dir(ground_truth_extr_pair, 0, 0, rebase=True, add_noise=False)
            gicp_extrinsics_pair, _ = helper_functions.add_noise_to_extr_dir(ground_truth_extr_pair, config.sigma_gicp_rot, config.sigma_gicp_trans, rebase=True, add_noise=config.add_noise_gicp_extr)
            
            gicp_extr_folder = os.path.join(output_path, "gicp_extr_" + str(config.sigma_gicp_rot) + "_" + str(config.sigma_gicp_trans))
            if not os.path.exists(gicp_extr_folder):
                os.mkdir(gicp_extr_folder)
            output_json_file(gicp_extr_folder, gt_extrinsics_pair, "gt_extrinsics_" + str(sensor_idx-1) + "_" + str(sensor_idx) + ".json")
            output_json_file(gicp_extr_folder, gicp_extrinsics_pair, "gicp_extrinsics_" + str(sensor_idx-1) + "_" + str(sensor_idx) + ".json")
            
            del ground_truth_extr_pair[str(sensor_idx-1)]
    
    global gt_extrinsics_all
    gt_extrinsics_all, _ = helper_functions.add_noise_to_extr_dir(ground_truth_extr_all, 0, 0, rebase=False, add_noise=False)
    
    
    assert(len(sensors) == len(camera_serials))
    
    return sensors

def output_json_file(output_folder, data, filename):
    config_json = os.path.join(output_folder, filename)
    with open(config_json, 'w') as f:
        json.dump(data, f, indent=4)

def parse_arguments(args):
    usage_text = (
        "Generate and compare synthetic dataset script."
        "Usage:  python synthetic_dataset.py [options],"
        "   with [options]: "
        "       -g, --generate (Runs in generate mode to create the synthetic dataset and the json file)"
        "       -o, --output_path=<relative path of all generate outputs>"
        "       -cf --config_file_path=<path of configuration file>"
        "       -c, --compare (Runs in compare mode to compare the extrinsics and intrinsics of calibration\' initials with the ground truth)" 
        "       -i, --initials_file=<path of calibration's initials output file (compare mode)>"
        "       -d, --dataset_info_file=<path of generate's json output file (compare mode)>" 
        
    )
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument("-g","--generate", action='store_true', help='Runs in generate mode to create the synthetic dataset and the json file')
    parser.add_argument("-c","--compare", action='store_true', help='Runs in compare mode to compare the extrinsics and intrinsics of calibration\' initials with the ground truth')

    parser.add_argument("-o","--output_path", type = str, help = "relative path of all generate outputs", default="results")
    parser.add_argument("-cf","--config_file_path", type = str, help = "path of configuration file", default="config.yml")
    parser.add_argument("-i","--initials_file", type = str, help = "path of calibration's initials output file (compare mode)", default="BA_Camera_Parameters_initial.json")
    parser.add_argument("-d","--dataset_info_file", type = str, help = "path of generate's json output file (compare mode)", default="dataset_info.json")

    return parser.parse_known_args(args)

def compare_multicamcalib(dataset_info_file, calib_initials_file):
    with open(dataset_info_file, 'r') as file:
        data = json.load(file)
        
    #FOR RAE
    plys_folder = os.path.join(os.path.dirname(dataset_info_file), "boxes/gicp")
    synthetic_points_for_camera_i = load_points_from_ply_folder(plys_folder)
     
    n_cams = data["calibration"]["n_cams"]   
    cam_serials = data["calibration"]["cam_serials"]
    
    with open(calib_initials_file, 'r') as file:
        calib = json.load(file)
    
    np.set_printoptions(precision=8, suppress=True)
    
    #(1) Always the same
    T_mi_cam_to_cv_cam = np.eye(4)
    T_mi_cam_to_cv_cam[0,0] = -1
    T_mi_cam_to_cv_cam[1,1] = -1

    #(2) Depends on dataset
    T_cv_w_to_mi_w = np.array([
                                [1, 0, 0, -0.2],
                                [0, 0, 1, 0],
                                [0,-1, 0, 0.32],
                                [0, 0, 0, 1]
                            ])
    
    #(2) If center cam is 4 in circle or line maybe
    if "circle\\no_rand\\ds_8_1.5m" in dataset_info_file:
        T_cv_w_to_mi_w = np.array([
                                    [-1, 0, 0, 0.2],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, -0.32],
                                    [0, 0, 0, 1]
                                ])
    #print(T_cv_w_to_mi_w @ [0,0,0,1])
    mean_angle = 0
    mean_t = 0
    mean_central = 0
    mean_focal = 0
    
    angle_errors = []
    t_errors = []
    
    output_lines = []
    all_errors_rae = []
    for i in range(n_cams):
        calib_data = calib[cam_serials[i]]
        print("Comparing extrinsics for camera " + cam_serials[i])
        
        r_gt = np.array(data["camera_to_world_mi_coords_"+cam_serials[i]]["rvec"])
        t_gt = np.array(data["camera_to_world_mi_coords_"+cam_serials[i]]["tvec"])
        
        T_mi_cam_to_mi_w = np.eye(4)  
        T_mi_cam_to_mi_w[:3,:3], _ = cv2.Rodrigues(np.float32(np.array(r_gt).flatten()))
        T_mi_cam_to_mi_w[:3,3] = np.float32(np.array(t_gt).flatten())
        
        r = np.array(calib_data["rvec"])
        t = np.array(calib_data["tvec"]) / 1000
        
        T_cv_w_to_cv_cam = np.eye(4)  
        T_cv_w_to_cv_cam[:3,:3], _ = cv2. Rodrigues(np.float32(np.array(r).flatten()))
        T_cv_w_to_cv_cam[:3,3] = np.float32(np.array(t).flatten())
        
        T_cv_cam_to_cv_w = np.linalg.inv(T_cv_w_to_cv_cam)
        
        T_gt_cv_cam_to_mi_w = T_mi_cam_to_mi_w @ T_mi_cam_to_cv_cam
        T_cv_cam_to_mi_w = T_cv_w_to_mi_w @ T_cv_cam_to_cv_w
        
        #print("EXTRINSICS")
        R_error = T_cv_cam_to_mi_w[:3,:3] @ T_gt_cv_cam_to_mi_w[:3,:3].T
        
        # FOR RAE ----
        pts_i_cam = synthetic_points_for_camera_i[i]  # shape (N_i,3)
        R_i_gt = T_gt_cv_cam_to_mi_w[:3,:3]
        T_i_gt = T_gt_cv_cam_to_mi_w[:3,3]
        R_i_est = T_cv_cam_to_mi_w[:3,:3]
        T_i_est = T_cv_cam_to_mi_w[:3,3]
        errs_i = compute_icp_pointwise_RAE(pts_i_cam, R_i_gt, T_i_gt, R_i_est, T_i_est)
        all_errors_rae.extend(errs_i)
        #-------------
        
        #The origin of the checkerboard is at [0,0,0] and is the center of the botom surface of the cube. Since its height is 6mm we do:
        T_gt_cv_cam_to_mi_w[1,3] = T_gt_cv_cam_to_mi_w[1,3] - 0.06
        
        rounded = round((np.trace(R_error)-1)/ 2,7)
        angle_error = np.arccos(rounded) * (180 / np.pi)
        t_error = np.linalg.norm(T_cv_cam_to_mi_w[:3,3] - T_gt_cv_cam_to_mi_w[:3,3])
        
        print(T_cv_cam_to_mi_w[:3,3])
        print(T_gt_cv_cam_to_mi_w[:3,3])
        
        t_error = t_error*1000
        
        mean_angle += angle_error
        mean_t += t_error
        
        t_errors.append(t_error)
        angle_errors.append(angle_error)
        
        print("angle_error=", angle_error)
        print("t_error=", t_error, "mm")
        print("-----------------------")
        
        cx_gt = np.array(data["camera"+ cam_serials[i] + "_intrinsics"]["cx"])
        cy_gt = np.array(data["camera"+ cam_serials[i] + "_intrinsics"]["cy"])
        fx_gt = np.array(data["camera"+ cam_serials[i] + "_intrinsics"]["fx"])
        fy_gt = np.array(data["camera"+ cam_serials[i] + "_intrinsics"]["fy"])
        
        cx = np.array(calib_data["cx"])
        cy = np.array(calib_data["cy"])
        fx = np.array(calib_data["fx"])
        fy = np.array(calib_data["fy"])
        
        central_distance = math.sqrt((cx_gt - cx) ** 2 + (cy_gt - cy) ** 2)
        focal_distance = math.sqrt((fx_gt - fx) ** 2 + (fy_gt - fy) ** 2)
        
        mean_central += central_distance
        mean_focal += focal_distance
        
    mean_angle_error = np.mean(angle_errors)
    min_angle_error = min(angle_errors)
    max_angle_error = max(angle_errors)
    median_angle_error = np.median(angle_errors)
    
    mean_t_error = np.mean(t_errors)
    min_t_error = min(t_errors)
    max_t_error = max(t_errors)
    median_t_error = np.median(t_errors)
    
    print("mean_angle=",mean_angle/n_cams)
    print("mean_t=",mean_t/n_cams)
    
    print("mean_central=",mean_central/n_cams)
    print("mean_focal=",mean_focal/n_cams)  
    
    # Compute statistics
    mean_error = np.mean(all_errors_rae)
    rms_error = np.sqrt(np.mean(np.square(all_errors_rae)))
    median_error = np.median(all_errors_rae)
    
    with open(os.path.join(os.path.dirname(dataset_info_file),"results/multicamcalib/multicamcalib_comparison_results.txt"), "w") as f:
        f.write(f"mean error translation (mm) : {mean_t_error}\n")
        f.write(f"mean error rotation (degree) : {mean_angle_error}\n")
        f.write(f"median error translation (mm) : {median_t_error}\n")
        f.write(f"median error rotation (degree) : {median_angle_error}\n")
        f.write(f"min error translation (mm) : {min_t_error}\n")
        f.write(f"min error rotation (degree) : {min_angle_error}\n")
        f.write(f"max error translation (mm) : {max_t_error}\n")
        f.write(f"max error rotation (degree) : {max_angle_error}\n")
        f.write(f"mean central : {mean_central/n_cams}\n")
        f.write(f"mean focal : {mean_focal/n_cams}\n")
        f.write(f"\nRAE estimation from pointclouds:\n")
        f.write(f"Mean Error: {mean_error:.4f} units\n")
        f.write(f"RMS Error: {rms_error:.4f} units\n")
        f.write(f"Median Error: {median_error:.4f} units\n")
    
    
def render_snapshot_dataset_images(sensors, camera_serials, dataset_folder):
    idx = 0
    my_scene = load_scene([0,0,0], [0,0,0], mode="boxes")
        
    gicp_folder = os.path.join(dataset_folder, "gicp/")
    if not os.path.exists(gicp_folder):
        os.mkdir(gicp_folder)
    
    for sensor_idx in range(len(sensors)):            
        image = mitsuba.render(scene=my_scene, spp=config.spp, sensor=sensors[sensor_idx])
        
        #----------------------Save ply---------------------
        #if(config.save_ply): #always save plys in the case of snapshot dataset
        my_scene_depth = load_scene_depth([0,0,0], [0,0,0], mode="boxes")
        aov_image = mitsuba.render(scene=my_scene_depth, spp=132, sensor=sensors[sensor_idx])
        
        #Filename for gicp
        img_filename = gicp_folder + str(idx) + "_" + camera_serials[sensor_idx] + "_color" + ".png"
        mitsuba.util.write_bitmap(img_filename, image)
        if(config.add_distortion):
            img = None
            while(img is None):
                img = cv2.imread(img_filename)
            img = helper_functions.distort_image_opencv(img, config.dist_coeffs)
            cv2.imwrite(img_filename, img)
        
        filename = gicp_folder + str(idx) + "_" + camera_serials[sensor_idx]
        helper_functions.save_ply(aov_image, config.downsample, config.voxel_size, filename, save_pcd=True, gt_extrinsics=gt_extrinsics_all[camera_serials[sensor_idx]], base=None, add_noise=config.add_pointcloud_noise, noise_std=config.noise_pointcloud_std) #saves pointclouds both as ply and pcd

        filename = gicp_folder + camera_serials[sensor_idx] + "_" + str(idx) + "_depth_mm.pgm"
        helper_functions.save_pgm(aov_image, sensors[sensor_idx], filename)

def generate(output_path):
    np.set_printoptions(precision=8, suppress=True)
    
    if(config.shape == 'rectangle' or config.shape == 'custom'):
        camera_angles = [0] * config.num_cameras
        print(camera_angles)
    else: 
        camera_angles = []
        angle = 0
        for i in range(config.num_cameras):
            camera_angles.append(angle)
            angle += 360/config.num_cameras
    
    camera_targets = helper_functions.generate_camera_targets(config.targets, config.sigma_targets, config.random_targets)
    
    print("Targets")
    print(camera_targets)
    print("--------------------------")
    
    if(config.shape == "custom"):
        camera_origins = config.origins
    else:
        camera_origins = helper_functions.generate_camera_origins(config.shape, config.d1, config.d2, config.heights, config.sigma_heights, config.random_heights)
    
    print("Origins")
    print(camera_origins)
    print("--------------------------")
    
    camera_serials = []
    for i in range(config.num_cameras):
        camera_serials.append(str(i))
    
    center_obj_pos = [0,0,0]
    
    sensors = load_sensors(camera_serials, camera_angles, config.center_cam_idx, camera_origins, camera_targets)

    obj_positions, obj_angles = helper_functions.generate_obj_positions_and_angles(config.shape, config.obj_d1, config.obj_d2)

    #---------------tmp for faster tests--------------------------------------
    #obj_angles_pos_1 = [[0,j,0] for j in [x for x in range(0,360,30)]]
    #obj_angles_pos_2 = [[90,j,0] for j in [x for x in range(0,360,10)]]
    
    #obj_angles = [obj_angles_pos_1, obj_angles_pos_2]    
    #obj_positions = [[0,0,0], [0,0.5,0]]
    
    #obj_angles = [[[0,0,0]]]
    #obj_positions = [[0,0,0]]
    #-------------------------------------------------------------------------
    if(config.generate_charuco_ds):
        dataset_folder = os.path.join(output_path, "charuco/")
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        global board_obj, board_texture
        board_obj = config.charuco_obj
        board_texture = config.charuco_texture
        render_dataset_images(sensors, obj_positions, obj_angles, config.center_cam_idx, center_obj_pos, camera_serials, dataset_folder, output_path)

    if(config.generate_checker_ds):
        dataset_folder = os.path.join(output_path, "checkerboard/")
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        board_obj = config.checker_obj
        board_texture = config.checker_texture
        render_dataset_images(sensors, obj_positions, obj_angles, config.center_cam_idx, center_obj_pos, camera_serials, dataset_folder, output_path) 
    
    if(config.generate_structNet_ds):
        dataset_folder = os.path.join(output_path, "boxes/")
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        render_snapshot_dataset_images(sensors, camera_serials, dataset_folder)
        #helper_functions.save_device_repository(camera_serials, output_path)
    
def main(argv):
    # Parse commandline arguments
    args, _ = parse_arguments(argv)
    
    global config, output_path
    config = Config(args.config_file_path)
    output_path = args.output_path
    
    if args.generate:
        #Copy config file to the output path to keep information about the dataset generation
        #shutil.copy2(args.config_file_path, args.output_path)
        
        generate(args.output_path)

        output_json_file(args.output_path, dataset_info, "dataset_info.json")
        output_json_file(args.output_path, gt_extrinsics_all, "gt_extrinsics_all.json")
    
    elif args.compare:
        compare_multicamcalib(args.dataset_info_file, args.initials_file)
    
    else:
        print(f"ERROR: Generate -g or Compare -c mode must be selected")
        exit(-1)

if __name__ == "__main__":
    main(sys.argv)
    exit()