import yaml

class Config:
    def __init__(self, config_filepath):
        with open(config_filepath,'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            
        self.spp = config['samples_per_pixel']
        self.save_ply = config['save_ply']
        
        self.add_noise_gicp_extr = False
        self.sigma_gicp_rot = 0
        self.sigma_gicp_trans = 0
        if 'sigma_gicp_rot' in config and 'sigma_gicp_trans' in config:   
            self.add_noise_gicp_extr = True
            self.sigma_gicp_rot = config['sigma_gicp_rot']
            self.sigma_gicp_trans = config['sigma_gicp_trans']
        
        #----------- Camera config -----------
        cam_config = config['camera_config']
        
        self.num_cameras = cam_config['num_cameras']
        self.center_cam_idx = cam_config['center_cam_idx']
        self.shape = cam_config['configuration_shape']
        
        if(self.shape == 'rectangle'):
            assert(self.num_cameras == 8)
        
        self.d1 = cam_config['d1']
        self.d2 = 0
        if 'd2' in cam_config:
            self.d2 = cam_config['d2']
            
        self.targets = cam_config['targets']
        assert(len(self.targets) == self.num_cameras)
        
        self.random_targets = False
        self.sigma_targets = 0
        if 'sigma_targets' in cam_config:
            self.random_targets = True
            self.sigma_targets = cam_config['sigma_targets']
            
        self.heights = cam_config['heights']
        assert(len(self.heights) == self.num_cameras)
        
        self.random_heights = False
        self.sigma_heights = 0
        if 'sigma_heights' in cam_config:
            self.random_heights = True
            self.sigma_heights = cam_config['sigma_heights']
            
        if(self.shape == 'custom'):
            assert('origins' in cam_config)
            self.origins = cam_config['origins']
            assert(len(self.origins) == self.num_cameras)
            
        self.random_origins = False
        self.sigma_origins = 0
        if 'sigma_origins' in cam_config:
            self.random_origins = True
            self.sigma_origins = cam_config['sigma_origins']
        
        #----------- Object config -----------
        obj_config = config['object_config']
        
        self.generate_checker_ds = False
        if('checker_obj_filepath' in obj_config):
            self.generate_checker_ds = True
            self.checker_obj = obj_config['checker_obj_filepath']
            self.checker_texture = obj_config['checker_texture_filepath']
        
        self.generate_charuco_ds = False
        if('charuco_obj_filepath' in obj_config):
            self.generate_charuco_ds = True
            self.charuco_obj = obj_config['charuco_obj_filepath']
            self.charuco_texture = obj_config['charuco_texture_filepath']
            
        self.generate_structNet_ds = False
        if('boxes_folder_path' in obj_config):
            self.generate_structNet_ds = True
            self.boxes = obj_config['boxes_folder_path']
        
        self.obj_d1 = obj_config['pos_cross_d1']
        self.obj_d2 = 0
        if 'pos_cross_d2' in obj_config:
            self.obj_d2 = obj_config['pos_cross_d2']
            
        #----------- Scene config ----------- 
        scene_config = config['scene_config']
        
        self.envmap = scene_config['envmap_filepath']
        self.envmap_scale = scene_config['envmap_light_scale']
        self.envmap_rot_axis = scene_config['envmap_rotation_axis']
        self.envmap_rot_degrees = scene_config['envmap_rotation_degrees']
        self.room = scene_config['room_folder_path']
        
        #----------- Post-processing config ----------- 
        
        post_proc_config = config['post_processing_config']
                
        self.add_distortion = post_proc_config['add_distortion']
        self.dist_coeffs = []
        if 'distortion_coefficients' in post_proc_config:
            self.dist_coeffs = post_proc_config['distortion_coefficients']
        
        self.downsample = post_proc_config['downsample']
        self.voxel_size = 0
        if 'voxel_size_downsampling' in post_proc_config:
            self.voxel_size = post_proc_config['voxel_size_downsampling']
            
        self.add_pointcloud_noise = False
        self.noise_pointcloud_std = 0
        if 'add_pointcloud_noise' in post_proc_config:
            self.add_pointcloud_noise = post_proc_config['add_pointcloud_noise']
            self.noise_pointcloud_std = post_proc_config['noise_pointcloud_std']
            
        self.add_noise = post_proc_config['add_noise']
        if 'noise_std' in post_proc_config:
            self.noise_std = post_proc_config['noise_std']
            