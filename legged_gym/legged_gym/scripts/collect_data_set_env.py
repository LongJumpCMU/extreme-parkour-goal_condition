import subprocess
import os
import time
import numpy as np
import datetime
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data_set

def get_coverage(robot_pos_x, robot_pos_y):
    max_valid_coverage = 1.42
    y_min, y_max, x_min, x_max = (
        robot_pos_y - max_valid_coverage,
        robot_pos_y + max_valid_coverage,
        robot_pos_x - max_valid_coverage,
        robot_pos_x + max_valid_coverage,
    )

    return y_min, y_max, x_min, x_max


def rotate_point(point, center, angle):
    # Extract coordinates
    x, y = point
    cx, cy = center

    # Perform the rotation
    new_x = cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    new_y = cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    

    return np.array([new_x, new_y])

def rotate_point_array(point, center, angle):
    # Extract coordinates
    y = point[:,0]
    y = point[:,1]
    cx, cy = center

    # Perform the rotation
    new_x = cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    new_y = cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)

    indices1, indices2 = np.meshgrid(np.arange(len(new_x)), np.arange(len(new_y)))
    paired_array = np.column_stack((new_x[indices1.ravel()], new_y[indices2.ravel()]))


    return paired_array#np.array([new_x, new_y])

def read_config(json_path):
    with open(json_path, 'r') as file:
        # Load the JSON data into a Python list
        config = json.load(file)
    return config

def scale_env(map, config):
    scaled_array = np.clip(map.astype(np.int16) * ((config["max_height"]-config["min_height"]) / 255.0)  + config["min_height"], config["min_wall_height"],config["max_wall_height"])
    # scaled_array = map.astype(np.int16) * ((config["max_height"]-config["min_height"]) / 255)  + config["min_height"]

    # import ipdb; ipdb.set_trace()
    return scaled_array

def pos2idx_array(pos, config):
    return (pos/config["horizontal_scale"]).astype(int)

def pos2idx(pos, config):
    return round(pos/config["horizontal_scale"])

def idx2pos(idx, config):
    return idx*config["horizontal_scale"]

def divide_env(map, scandotsx, scandotsy,config):
    size_scandotx = pos2idx(abs(scandotsx[-1]-scandotsx[0]),config)
    size_scandoty = pos2idx(abs(scandotsy[-1]-scandotsy[0]),config)

    len_x = round(map.shape[0]/size_scandotx)
    len_y = round(map.shape[1]/size_scandoty)

    incrementx = pos2idx(abs(scandotsx[0]),config)
    incrementy = pos2idx(abs(scandotsy[0]),config)
    start_pointsx = np.zeros(len_x)
    start_pointsy = np.zeros(len_y)

    for x in range(0,len_x):
        start_pointsx[x] = (x)*size_scandotx+incrementx
    for y in range(0,len_y):
        start_pointsy[y] = (y)*size_scandoty+incrementy


    
    return start_pointsx,start_pointsy


def is_within_height_limit(height_map, x, y, max_height, config):
    """
    Checks if the height at the specified point (x, y) is within the specified limit.
    """
    x_idx = pos2idx(x,config)
    y_idx = pos2idx(y,config)
    return height_map[x_idx,y_idx] <= max_height and height_map[x_idx,y_idx] >= config["min_ditch_spawn"]

def is_distance_safe(height_map, target_x, target_y, radius, max_height,config):
    # Define the bounds for the search area
    min_x = pos2idx(max(0, target_x - radius),config)
    max_x = min(height_map.shape[0] - 1, pos2idx(target_x + radius,config))
    min_y = pos2idx(max(0, target_y - radius),config)
    max_y = min(height_map.shape[1] - 1, pos2idx(target_y + radius,config))
    
    # Iterate over the search area
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # try:
            if height_map[x, y] >= max_height:
                distance = np.sqrt((target_x - idx2pos(x, config))**2 + (target_y - idx2pos(y,config))**2)
                if distance < radius:
                    return False
            # except:
            #     import ipdb; ipdb.set_trace()
    return True

def is_wall_between_points(height_map, start_x, start_y, target_x, target_y, max_wall_height,config):
    """
    Checks if there is a wall between the start and target points based on the specified maximum wall height.
    """
    # Calculate the slope of the line between the start and target points
    dx = target_x - start_x
    dy = target_y - start_y

    # Iterate over the line and check if any point exceeds the maximum wall height
    steps = pos2idx(max(abs(dx), abs(dy)),config)*5
    for i in range(steps + 1):
        x = pos2idx(start_x + i * dx / steps,config)
        y = pos2idx(start_y + i * dy / steps,config)
        if height_map[x, y] >= max_wall_height:
            return True
    return False

def on_close_wall_valid(robot_pos_x, robot_pos_y, target_pos_x, target_pos_y, config, map):
    if is_within_height_limit(map, robot_pos_x,robot_pos_y,np.max(map), config) \
        and is_distance_safe(map, robot_pos_x,robot_pos_y,config["robot_clearance"],np.max(map),config) \
        and is_within_height_limit(map, target_pos_x,target_pos_y,np.max(map), config) \
        and is_distance_safe(map, target_pos_x,target_pos_y,config["robot_clearance"],np.max(map),config) \
        and not is_wall_between_points(map, robot_pos_x,robot_pos_y,target_pos_x,target_pos_y,np.max(map),config): # make sure it is not a wall, or close to a wall, or a wall in between start and target point
        
        return True
    
    else:
        return False

def bounds_valid(map, target_pos_x, target_pos_y,config):
    if pos2idx(target_pos_x,config) >= map.shape[0] or pos2idx(target_pos_y,config) >= map.shape[1]:
        return False
    else:
        return True
        

def divide_heading(num_heading, mode = 'FRONT'):
    heading = None
    if mode == 'FRONT':
        heading = np.linspace(-np.pi/2, np.pi/2, num=num_heading)
    
    return heading

def valid_waypoint(
    robot_pos_x,
    robot_pos_y,
    target_pos_x,
    target_pos_y,
    scandots_range,
    config, 
    map
):
    # need to be able to: no target point for walls/nearby walls, outside of bounds, 5 heading angles, beaware of termination conditions, points past walls, no points behind and around a parameter around the robot
    # keep in mind that now the robot pos will be abolute position and target will also be absolute?

    # transform_matrix = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    robot_pos = np.array([robot_pos_x, robot_pos_y])
    target_pos = np.array([target_pos_x, target_pos_y])
    # transformed_target_pos = rotate_point(target_pos, robot_pos, -heading) # need to do this in main lopp!!!!!!!!!!!!!!!!!!!!!!!!!SS
    target_distance = np.linalg.norm(robot_pos-target_pos)
    # y_min, y_max, x_min, x_max = get_coverage(robot_pos_x, robot_pos_y)

    if (
        target_pos[0] >= scandots_range[0][0] + robot_pos_x
        and target_pos[0] <= scandots_range[0][1] + robot_pos_x
        and target_pos[1] >= scandots_range[1][0] + robot_pos_y
        and target_pos[1] <= scandots_range[1][1] + robot_pos_y
    ):
        # form a parmeter around the robot's com so that they won't get used
        # make sure the point is not on a wall and a good distance from the wall (ie. half of robot's length and more)
        # make sure the robot is not out of bounds
        # print("scandots)
        if target_distance >= config["min_target_dis"] \
            and bounds_valid(map,target_pos_x,target_pos_y, config) \
            and bounds_valid(map,robot_pos_x,robot_pos_y, config):
            # and on_close_wall_valid(robot_pos_x,robot_pos_y,target_pos_x,target_pos_y,config,map):
            
            return True
        else:
            return False
    else:
        return False

def check_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If not, create the directory
        os.makedirs(directory_path)

def plot_final_points(all_start, all_targets, height_map, dataset_config):
    all_start = np.array(all_start)
    all_targets = np.array(all_targets)
    with open('start.npy', 'wb') as f:
        np.save(f, all_start)

    with open('target.npy', 'wb') as f:
        np.save(f, all_targets)

    plt.figure()
    
    # plt.imshow(mpimg.imread(dataset_config["img_path"]))
    axis_1 = np.linspace(0, height_map.shape[1] - 1, height_map.shape[1])#/dataset_config['horizontal_scale']  # axis_1 coordinates
    axis_0 = np.linspace(0, height_map.shape[0] - 1, height_map.shape[0])#//dataset_config['horizontal_scale']  # axis_0 coordinates
    heightmap = plt.pcolormesh(axis_1, axis_0, height_map, cmap="viridis")
    plt.colorbar(label="Height")
    


    # plt.scatter([5],[5],label='random point')
    plt.scatter(pos2idx_array(all_start[:,1],dataset_config),pos2idx_array(all_start[:,0],dataset_config),label='starting points')
    plt.scatter(pos2idx_array(all_targets[:,1],dataset_config),pos2idx_array(all_targets[:,0],dataset_config),label='target points')
    plt.xlabel("Axis = 1")
    plt.ylabel("Axis = 0")

    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

def get_resume_idx(dataset_config):
    if os.path.isfile(dataset_config["data_file"]):
        all_data = data_set.TestDataset(dataset_config["data_file"])
        return len(all_data.data) - 1
    else:
        return 0
def all_valid_pnts(scandots_x,scandots_y,SCANDOTS_RANGE,dataset_config, height_map, heading, verbose=False):
    
    starting_listx,starting_listy = divide_env(height_map,scandots_x,scandots_y,dataset_config)
    starting_listx = idx2pos(np.array(starting_listx),dataset_config)
    starting_listy = idx2pos(np.array(starting_listy),dataset_config)
    heading_list = heading
    start_idx = get_resume_idx(dataset_config)
    

    granularity = 1.0
    itr = 0
    hashset = set()
    resume_itr = 1
    str_granularity = f"granularity:{granularity}"
    LAST_COMPLETED_ITR = itr

    # Obstacle height [0, 0.7]/depth [-0.1, -0.4]
    # print("granularity: ", granularity)
    all_start = []
    all_targets = []
    all_valid_start = []
    all_valid_pairs = []
    for starting_y_granularity in starting_listy:
        for starting_x_granularity in starting_listx:
            
            # Waypoint x [0, 1]
            for target_x_granularity in scandots_x:
                
                # Waypoint y [-1, 1]
                target = [0,0]
                for target_y_granularity in scandots_y:
                    target[1] = target_y_granularity + starting_y_granularity

                    for heading_granularity in heading_list:
                        # import ipdb; ipdb.set_trace()
                        start = [starting_x_granularity,starting_y_granularity, heading_granularity]
                        all_start.append(start)
                        #######################################
                        target[0] = target_x_granularity + starting_x_granularity
                        # rotate the target
                        transformed_target_pos = rotate_point(target, start[0:2], heading_granularity)
                        # import ipdb; ipdb.set_trace()
                        if not valid_waypoint(
                            starting_x_granularity,
                            starting_y_granularity,
                            transformed_target_pos[0],
                            transformed_target_pos[1],
                            SCANDOTS_RANGE,
                            dataset_config,
                            height_map
                        ):
                            print("not valid")
                            
                            continue

                        print("valid point!!!!!!!!!!!!!!!!")
                        all_targets.append(transformed_target_pos)
                        start = [starting_x_granularity,starting_y_granularity, heading_granularity]

                        transformed_target_pos = transformed_target_pos.tolist()
                        all_valid_pairs.append(start+transformed_target_pos)


                        itr += 1

    if verbose == True:
        plot_final_points(np.array(all_valid_pairs)[:,0:2], np.array(all_valid_pairs)[start_idx:-1,3:5], height_map, dataset_config)
    
    print("Numebr of datapoints already collected: ", start_idx)
    return all_valid_pairs[start_idx:-1]

def prepare_env(config_path):
    dataset_config = read_config(config_path)

    ##### Parameters #####
    # scale the environment
    image = Image.open(dataset_config["img_path"])
    image_array = np.array(image)
    height_map = scale_env(image_array, dataset_config)

    # Obstacle parameters
    scandots_x = dataset_config["scandots_axis0"]#[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
    scandots_y = dataset_config["scandots_axis1"]#[-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
    return scandots_x,scandots_y,dataset_config,height_map

def main(args):

    # First go to directory
    write_file_name = "dataset"
    current_directory = os.getcwd()
    print(current_directory)
    
    dataset_config = read_config(args.config_path)

    ##### Parameters #####
    # scale the environment
    image = Image.open(dataset_config["img_path"])
    image_array = np.array(image)
    height_map = scale_env(image_array, dataset_config)

    # Obstacle parameters
    scandots_x = dataset_config["scandots_axis0"]#[-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
    scandots_y = dataset_config["scandots_axis1"]#[-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]

    

    scandots_x,scandots_y,dataset_config, height_map = prepare_env(args.config_path)

    SCANDOTS_RANGE = [[scandots_x[0], scandots_x[-1]], [scandots_y[0], scandots_y[-1]]]

    


    ###################### RESUMING TRAINING #########################
    log_file_name = "completed_experiments.txt"
    check_directory(dataset_config["saved_data_dir"])
    log_path = os.path.join(dataset_config["saved_data_dir"], log_file_name)
    
    if not os.path.exists(log_path):
        line = f.readline()
                
    ##################################################################
    heading_list = divide_heading(dataset_config["heading_divide"]) #[np.pi/3]
    if not dataset_config["collect_with_planner"]:
        all_valid_pairs = np.array(all_valid_pnts(scandots_x,scandots_y,SCANDOTS_RANGE,dataset_config, height_map, heading_list, verbose=True))
    all_start = all_valid_pairs[:,0:3]
    all_target = all_valid_pairs[:,3:]

    termination = False
    start_idx = 0
    next_start = all_start[start_idx]
    next_target = np.array([all_target[start_idx].tolist()])
    # import ipdb;ipdb.set_trace()


    if args.run_type == 'unittest_validity':
        all_valid_pnts(scandots_x,scandots_y,SCANDOTS_RANGE,dataset_config, height_map, heading_list, verbose=True)
    elif args.run_type == 'patch_data_collection':
        process = subprocess.Popen(
                            [
                                "python",
                                "play_dataset_creation.py",
                                "--exptid",
                                # "000-16-go1_p_40_d_1_official_torque_run2_backup",
                                # "Mar-17-go1_A1limits_last_checkpoint",
                                "003-82",
                                "--task",
                                "dataset_go1",
                                "--num_itr",
                                str(0),
                                "--start",
                                str(next_start[0]), str(next_start[1]), str(next_start[2]),
                                "--target",
                                str(next_target[0,0]),str(next_target[0,1]),
                                "--data_file",
                                str(write_file_name),
                                "--device",
                                "cuda",
                                "--num_envs",
                                str(10),
                                "--headless",
                            ]
                        )
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--config_path', type=str, default='data_collection_configs/default_configs.json', help='the path for the config file')
    parser.add_argument('--run_type', type=str, default='patch_data_collection', help='ttype of run, can be unittest_validity, or search_data_collection, or patch_data_collection')
    
    args = parser.parse_args()

    main(args)