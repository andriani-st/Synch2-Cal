import mitsuba
import os

def create_object_dict_with_texture(obj_filename, texture_filename):
    object_dict = {
        'type': "obj",
        'filename': obj_filename,
        'bsdf': {
            "type": "diffuse",
                "reflectance": {
                "type": "bitmap",
                "filename": texture_filename,
                "wrap_mode": "clamp"
            }
        }
    }
    
    return object_dict

def create_object_dict(obj_filename, type = "diffuse", rgb = [1,1,1]):
    object_dict = {
        'type': "obj",
        'filename': obj_filename,
    }
    
    if(type == "diffuse"):
        object_dict["bsdf"] = {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": rgb
            }
        }
    elif(type == "dielectric"):
        object_dict["bsdf"] = {
            "type": "dielectric",
            "int_ior": "water",
            "ext_ior": "air"
        }
    
    return object_dict

def read_object_folder(root_dir):
    object_folders = [root_dir]
    for foldername, subdirs, _ in os.walk(root_dir):
        for subdir in subdirs:
                object_folders.append(os.path.join(foldername, subdir))

    texture_obj_tuples = []

    for object_folder in object_folders:
        obj_files = [f for f in os.listdir(object_folder) if f.endswith('.obj')]
        texture_files = [f for f in os.listdir(object_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for obj_file in obj_files:
            base_name = os.path.splitext(obj_file)[0]

            # Find the corresponding texture file (with the same name)
            matching_texture = ""
            for texture in texture_files:
                if os.path.splitext(texture)[0] == base_name:
                    matching_texture = texture
                    break
                
            if(matching_texture != ""):
                texture_obj_tuples.append((os.path.join(object_folder, obj_file),os.path.join(object_folder, matching_texture)))
            else:
                texture_obj_tuples.append((os.path.join(object_folder, obj_file),""))
                    
    return texture_obj_tuples
            
            
def get_room_objects(root_dir):
    texture_obj_tuples = read_object_folder(root_dir)
    
    obj_dict = {}
    for obj_file, texture_file in texture_obj_tuples:
        if(texture_file != ""):
            item = create_object_dict_with_texture(obj_file, texture_file)
        else:
            if("dielectric" in obj_file):
                item = create_object_dict(obj_file, "dielectric")
            else:
                item = create_object_dict(obj_file)
            
        base_name = os.path.splitext(os.path.basename(obj_file))[0]
        
        obj_dict[base_name] = item
        
    return obj_dict
        