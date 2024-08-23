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
import itertools
import random
import time



NUM_REGIONS = 0

def compute_yaw_single(cur_goals, root_states): # yaw should be bounded between -pi and pi!!!!!!!!!!!!!!!!!!!!!!
    # if env_ids.shape[0]>1:
    #     target_pos_rel = cur_goals[:2] - root_states[env_ids, :2]
    # else:

    target_pos_rel = cur_goals[:2] - root_states[:2]

    norm = np.linalg.norm(target_pos_rel)
    target_vec_norm = target_pos_rel / (norm + 1e-5)

    target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])

    return target_yaw

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

def height2map(height, config):
    return (height-config["min_height"])/(config["max_height"]-config["min_height"])*255
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
    # import ipdb; ipdb.set_trace()
    if is_within_height_limit(map, robot_pos_x,robot_pos_y,config["max_collision_height"], config) \
        and is_distance_safe(map, robot_pos_x,robot_pos_y,config["robot_clearance"],config["max_collision_height"],config) \
        and is_within_height_limit(map, target_pos_x,target_pos_y,config["max_collision_height"], config) \
        and is_distance_safe(map, target_pos_x,target_pos_y,config["robot_clearance"],config["max_collision_height"],config) \
        and not is_wall_between_points(map, robot_pos_x,robot_pos_y,target_pos_x,target_pos_y,config["max_collision_height"],config): # make sure it is not a wall, or close to a wall, or a wall in between start and target point
        
        return True
    
    else:
        return False

def bounds_valid(map, target_pos_x, target_pos_y,config):
    if pos2idx(target_pos_x,config) >= map.shape[0] or pos2idx(target_pos_y,config) >= map.shape[1] or pos2idx(target_pos_x,config) <0 or pos2idx(target_pos_y,config) < 0:
        return False
    else:
        return True
        

def divide_heading(num_heading, mode = 'ALL'):
    heading = None
    if mode == 'FRONT':
        heading = np.linspace(-np.pi/2, np.pi/2, num=num_heading-1)
    else:
        heading = np.linspace(np.pi/3, 2*(np.pi+np.pi/3), num=num_heading-1)
    
    return heading

def random_angle(start_angle, end_angle):
    return random.uniform(start_angle, end_angle)

def generate_heading_list_random(heading_list):
    random_array = np.random.rand(len(heading_list))
    return np.array(heading_list)+random_array

def valid_waypoint(
    robot_pos_x,
    robot_pos_y,
    target_pos_x,
    target_pos_y,
    scandots_range,
    config, 
    map, 
    reset_pnt = False
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
    ) or not reset_pnt:
        # form a parmeter around the robot's com so that they won't get used
        # make sure the point is not on a wall and a good distance from the wall (ie. half of robot's length and more)
        # make sure the robot is not out of bounds
        # print("scandots)
        if target_distance >= config["min_target_dis"] \
            and bounds_valid(map,target_pos_x,target_pos_y, config) \
            and bounds_valid(map,robot_pos_x,robot_pos_y, config):
            # and on_close_wall_valid(robot_pos_x,robot_pos_y,target_pos_x,target_pos_y,config,map):
            if (reset_pnt and not on_close_wall_valid(robot_pos_x,robot_pos_y,target_pos_x,target_pos_y,config,map)):
                return False
            return True
        else:
            return False
    else:
        return False

def random_sample_validity(x, y, centrex, centrey, min_distance, max_distance):
    distance = np.sqrt((x-centrex)**2+(y-centrey)**2)
    if distance < min_distance or distance > max_distance:
        return False
    return True

def check_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If not, create the directory
        os.makedirs(directory_path)

def plot_final_points(all_true_start, all_start, all_targets, height_map, dataset_config):
    all_start = np.array(all_start)
    all_targets = np.array(all_targets)
    all_true_start = np.array(all_true_start)

    with open('start.npy', 'wb') as f:
        np.save(f, all_start)

    with open('target.npy', 'wb') as f:
        np.save(f, all_targets)

    with open('true_start.npy', 'wb') as f:
        np.save(f, all_true_start)

    plt.figure()
    
    # plt.imshow(mpimg.imread(dataset_config["img_path"]))
    axis_1 = np.linspace(0, height_map.shape[1] - 1, height_map.shape[1])#/dataset_config['horizontal_scale']  # axis_1 coordinates
    axis_0 = np.linspace(0, height_map.shape[0] - 1, height_map.shape[0])#//dataset_config['horizontal_scale']  # axis_0 coordinates
    heightmap = plt.pcolormesh(axis_1, axis_0, height_map, cmap="viridis")
    plt.colorbar(label="Height")

    # plt.scatter([5],[5],label='random point')
    # plt.scatter(pos2idx_array(all_start[:,1],dataset_config),pos2idx_array(all_start[:,0],dataset_config),label='starting points')
    plt.scatter(pos2idx_array(all_targets[:,1],dataset_config),pos2idx_array(all_targets[:,0],dataset_config),label='target points')
    plt.scatter(pos2idx_array(all_start[:,1],dataset_config),pos2idx_array(all_start[:,0],dataset_config),label='starting points')
    plt.scatter(pos2idx_array(all_true_start[:,1],dataset_config),pos2idx_array(all_true_start[:,0],dataset_config),label='true_start points',c="black")


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
    
def circular_start(distance, start, angle):
    point = rotate_point(start+np.array([0,distance]), start, angle)

    return point
def all_reset_pnts(distance, start, heading_list, scandot_range, dataset_config, height_map):
    reset_points = []
    for heading_granularity in heading_list:
        reset_pos = circular_start(distance, start, heading_granularity)
        if not valid_waypoint(
                            reset_pos[0],
                            reset_pos[1],
                            start[0],
                            start[1],
                            scandot_range,
                            dataset_config,
                            height_map, 
                            reset_pnt=True
                        ):
                            print("not valid")
                            
                            continue
        reset_points.append(reset_pos)
    return reset_points

def get_block_start(dataset_config, block_type=0): # 0 for block, 1 for hurdle
    robot_len = dataset_config["robot_length"]
    padding = dataset_config["block_gap_padding"]*robot_len
    block_width = dataset_config["block_width"]*robot_len

    if dataset_config["start_option"]==3: # hurdle
        block_length = dataset_config["hurdle_length"]*robot_len
    else:
        block_length = dataset_config["block_length"]*robot_len
    robot_clearance = dataset_config["start_distance"]
    terrain_length = dataset_config["terrain_length"]
    terrain_width = dataset_config["terrain_width"]
    startx = []
    starty = []

    
    region_width = padding*2+block_width
    region_length = padding*2+block_length

    left_edge_pnt = [[padding+block_length+robot_clearance, padding+robot_len/2],[padding+robot_clearance, padding+robot_len/2]] #below and on block for first env, later ones just need to add offset
    right_edge_pnt = [[padding+block_length+robot_clearance, padding+block_width-robot_len/2],[padding+robot_clearance, padding+block_width-robot_len/2]] 
    middle_pnt = [[padding+block_length+robot_clearance, padding+block_width/2],[padding+robot_clearance, padding+block_width/2]] 

    single_env_pnts = np.array([left_edge_pnt,middle_pnt,right_edge_pnt])
    
    for i in range(terrain_length):
        for j in range(terrain_width):
            for k in range(single_env_pnts.shape[0]):
                # off block
                startx.append(single_env_pnts[k][0][0]+i*region_length)
                starty.append(single_env_pnts[k][0][1]+j*region_width)
                if dataset_config["start_option"]==1: # block
                    # on block
                    startx.append(single_env_pnts[k][1][0]+i*region_length)
                    starty.append(single_env_pnts[k][1][1]+j*region_width)

    # import ipdb;ipdb.set_trace()
    
    return startx,starty

def get_gap_start(dataset_config):
    return
def all_valid_pnts(scandots_x,scandots_y,SCANDOTS_RANGE,dataset_config, height_map, heading, verbose=False):
    global NUM_REGIONS
    
    if dataset_config["start_option"]==0:
        starting_listx,starting_listy = divide_env(height_map,scandots_x,scandots_y,dataset_config)
        starting_listx = idx2pos(np.array(starting_listx),dataset_config)
        starting_listy = idx2pos(np.array(starting_listy),dataset_config)
    elif dataset_config["start_option"]==1: # block
        starting_listx,starting_listy = get_block_start(dataset_config)
    elif dataset_config["start_option"]==2:
        starting_listx,starting_listy = get_gap_start(dataset_config)
    elif dataset_config["start_option"]==3: # hurdle
        starting_listx,starting_listy = get_block_start(dataset_config, block_type=1)

    
    
    heading_list = generate_heading_list_random(heading)
    start_idx = get_resume_idx(dataset_config)
    NUM_REGIONS = len(starting_listx)
    

    granularity = 1.0
    itr = 0
    hashset = set()
    resume_itr = 1
    str_granularity = f"granularity:{granularity}"
    LAST_COMPLETED_ITR = itr
    circular_distance = dataset_config["start_distance"]


    # Obstacle height [0, 0.7]/depth [-0.1, -0.4]
    # print("granularity: ", granularity)
    all_start = []
    all_targets = []
    all_valid_start = []
    # all_valid_pairs should be a list of np arrays, where each array represents a region, and each point in each region consists of starting points (circular rule) and end points which is the actual start + target (ie. [3:7])
    all_valid_pairs = []
    valid_pair_region = []
    index_region = 0
    
    current_idx = 0
    y_idx = 0
    for starting_y_granularity in starting_listy:
        y_idx += 1
        if dataset_config["start_option"] == 0:
            listx = starting_listx
        else:
            listx = [starting_listx[y_idx-1]]
        for starting_x_granularity in listx:
            current_idx+=1
            if len(valid_pair_region)!=0:
                all_valid_pairs.append(valid_pair_region)
                
            valid_pair_region = []
            # Waypoint x [0, 1]
            
            start = np.array([starting_x_granularity,starting_y_granularity])
            reset_pnts = all_reset_pnts(circular_distance, start, heading_list, SCANDOTS_RANGE, dataset_config, height_map)
            num_valid_resets = len(reset_pnts)
            print("num_valid_resets is: ", num_valid_resets)
            count_valid_targets = 0
                
            while count_valid_targets < dataset_config["sample_per_region"] and num_valid_resets != 0:
                for reset_pnt in reset_pnts:
                        transformed_target_pos = np.zeros(2)
                        transformed_target_pos[0] = random.uniform(starting_x_granularity - abs(scandots_x[0]), starting_x_granularity + abs(scandots_x[-1]))
                        transformed_target_pos[1] = random.uniform(starting_y_granularity - abs(scandots_y[0]), starting_y_granularity + abs(scandots_y[-1]))
                        starting_point = np.array([starting_x_granularity,starting_y_granularity])
                        transformed_target_pos = rotate_point(transformed_target_pos, start[0:2], compute_yaw_single(starting_point, np.array([reset_pnt[0], reset_pnt[1]])))
                        while not (valid_waypoint(
                            starting_x_granularity,
                            starting_y_granularity,
                            transformed_target_pos[0],
                            transformed_target_pos[1],
                            SCANDOTS_RANGE,
                            dataset_config,
                            height_map
                        ) and random_sample_validity(
                            transformed_target_pos[0],
                            transformed_target_pos[1], 
                            starting_x_granularity, 
                            starting_y_granularity, 
                            dataset_config["start_distance"], 
                            abs(scandots_x[-1] - scandots_x[0]))):
                            print("not valid, keep trying")
                            # randomly select points until it is valid
                            # import ipdb;ipdb.set_trace()
                            transformed_target_pos[0] = random.uniform(starting_x_granularity - abs(scandots_x[-1] - scandots_x[0])/2, starting_x_granularity + abs(scandots_x[-1] - scandots_x[0])/2)
                            transformed_target_pos[1] = random.uniform(starting_y_granularity - abs(scandots_y[-1] - scandots_y[0])/2, starting_y_granularity + abs(scandots_y[-1] - scandots_y[0])/2)
                            transformed_target_pos = rotate_point(transformed_target_pos, start[0:2], compute_yaw_single(starting_point, np.array([reset_pnt[0], reset_pnt[1]])))

                        print("now is valid!!")
                        
                        transformed_target_pos = transformed_target_pos.tolist()
                        if count_valid_targets < dataset_config["sample_per_region"]:
                            valid_pair_region.append(reset_pnt.tolist()+start.tolist()+transformed_target_pos)
                        count_valid_targets+=1
                        print("num of valid: ", count_valid_targets, "at starting point: ", current_idx, "total is: ", len(starting_listy))
                        # import ipdb;ipdb.set_trace()

                        itr += 1
        index_region+=1
    # import ipdb;ipdb.set_trace()
    
    if verbose == True:
        merged = list(itertools.chain(*all_valid_pairs))
        plot_final_points(np.array(merged)[:,0:2], np.array(merged)[:,2:4], np.array(merged)[:,4:6], height_map, dataset_config)
    
    print("Numebr of datapoints already collected: ", start_idx)
    # return all_valid_pairs[start_idx:-1]
    # import ipdb; ipdb.set_trace()
    return all_valid_pairs


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
    robot_length = dataset_config["robot_length"]
    patch_x_range = dataset_config["patch_axis0_range"]
    patch_y_range = dataset_config["patch_axis1_range"]
    patch_resolution = dataset_config["patch_resolution"]
    patch_x = np.arange(patch_x_range[0]*robot_length, patch_x_range[1]*robot_length, patch_resolution)
    patch_y = np.arange(patch_y_range[0]*robot_length, patch_y_range[1]*robot_length, patch_resolution)
    # import ipdb; ipdb.set_trace()

    return scandots_x,scandots_y,dataset_config,height_map, patch_x, patch_y

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

    

    scandots_x,scandots_y,dataset_config, height_map, patchx,patchy = prepare_env(args.config_path)

    SCANDOTS_RANGE = [[patchx[0], patchx[-1]], [patchy[0], patchy[-1]]]
    num_agents = 10
    total_envs = NUM_REGIONS*num_agents

    
    


    ###################### RESUMING TRAINING #########################
    log_file_name = "completed_experiments.txt"
    check_directory(dataset_config["saved_data_dir"])
    log_path = os.path.join(dataset_config["saved_data_dir"], log_file_name)
    
    if not os.path.exists(log_path):
        line = f.readline()
                
    ##################################################################
    heading_list = divide_heading(dataset_config["heading_divide"]) #[np.pi/3]
    if not dataset_config["collect_with_planner"]:# and dataset_config["start_option"]==0:
        all_valid_pairs = np.array(all_valid_pnts(patchx,patchy,SCANDOTS_RANGE,dataset_config, height_map, heading_list))
        num_regions = all_valid_pairs.shape[0]
        total_envs = num_regions*num_agents
        # import ipdb;ipdb.set_trace()
    # elif dataset_config["start_option"]==1:
    #     starting_listx,starting_listy = get_block_start(dataset_config)
    #     num_regions = len(starting_listx)
    #     import ipdb;ipdb.set_trace()
    #     total_envs = num_regions*num_agents
    # elif dataset_config["start_option"]==2:
    #     starting_listx,starting_listy = get_gap_start(dataset_config)
    #     num_regions = len(starting_listx)
    #     total_envs = num_regions*num_agents

    
    # all_start = all_valid_pairs[:,0:3]
    # all_target = all_valid_pairs[:,3:]

    # termination = False
    # start_idx = 0
    # next_start = all_start[start_idx]
    # next_target = np.array([all_target[start_idx].tolist()])
    # import ipdb;ipdb.set_trace()



    if args.run_type == 'unittest_validity':
        all_valid_pnts(patchx,patchy,SCANDOTS_RANGE,dataset_config, height_map, heading_list, verbose=True)
    elif args.run_type == 'patch_data_collection':
        process = subprocess.Popen(
                            [
                                "python",
                                "play_dataset_creation.py",
                                "--exptid",
                                # "000-16-go1_p_40_d_1_official_torque_run2_backup",
                                # "Mar-17-go1_A1limits_last_checkpoint",
                                "000-82",
                                "--task",
                                "dataset_go1",
                                "--num_itr",
                                str(0),
                                # "--start",
                                # str(next_start[0]), str(next_start[1]), str(next_start[2]),
                                # "--target",
                                # str(next_target[0,0]),str(next_target[0,1]),
                                "--data_file",
                                str(write_file_name),
                                "--device",
                                "cuda",
                                "--num_envs",
                                str(total_envs),
                                "--num_regions", 
                                str(num_regions),
                                "--num_agents",
                                str(num_agents),
                                
                                "--headless",
                                "--config_path",
                                args.config_path,
                            ]
                        )
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--config_path', type=str, default='data_collection_configs/default_configs.json', help='the path for the config file')
    parser.add_argument('--run_type', type=str, default='patch_data_collection', help='ttype of run, can be unittest_validity, or search_data_collection, or patch_data_collection')
    
    args = parser.parse_args()

    start = time.time()

    main(args)

    end = time.time()
    print("total time elapsed for data collection is: ", end - start)