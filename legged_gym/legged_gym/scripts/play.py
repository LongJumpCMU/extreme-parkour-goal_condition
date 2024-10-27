# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from collect_data_set_env import prepare_env, all_valid_pnts, check_directory, valid_waypoint, bounds_valid, on_close_wall_valid, is_wall_between_points, is_distance_safe, is_within_height_limit, read_config, scale_env, pos2idx,pos2idx_array, idx2pos, rotate_point, divide_heading
import json
import csv
from pathlib import Path


def read_planner_config(json_path):
    with open(json_path, 'r') as file:
        # Load the JSON data into a Python list
        config = json.load(file)
    return config
def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


base_dir = "../../../../planning-project/results/"

def create_gym_run_log(exp_dir) -> str:
    top_dir = os.path.join(base_dir, "gym_log.csv")
    
    if not Path(top_dir).exists():
        try:
            os.makedirs(top_dir, exist_ok=True)
            print(f"Directories created: {top_dir}")
        except Exception as e:
            print(f"Failed to create directories: {e}")
    else:
        print(f"Directory already exists: {top_dir}")
    
    mid_dir = os.path.join(top_dir, env_type)
    if not Path(mid_dir).exists():
        try:
            os.makedirs(mid_dir, exist_ok=True)
            print(f"Directories created: {mid_dir}")
        except Exception as e:
            print(f"Failed to create directories: {e}")
    else:
        print(f"Directory already exists: {mid_dir}")
    
    exp_dir = os.path.join(mid_dir, exp_type)
    if not Path(exp_dir).exists():
        try:
            os.makedirs(exp_dir, exist_ok=True)
            print(f"Directories created: {exp_dir}")
        except Exception as e:
            print(f"Failed to create directories: {e}")
    else:
        print(f"Directory already exists: {exp_dir}")
    
    return exp_dir


def log_data(data, data_file_name,exp_dir):
    # import ipdb; ipdb.set_trace()
    filepath = os.path.join(exp_dir, data_file_name)
    if os.path.isfile(filepath):
            with open(filepath, "a", newline='') as f:
                writer = csv.writer(f)
                data_row = [data['success'], data['energy_cost'], data['time_cost']]
                writer.writerow(data_row)
                # f.write(data)
    else:
        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            header = ['success', 'energy_cost', 'time_cost']
            writer.writerow(header)
            data_row = [data['success'], data['energy_cost'], data['time_cost']]
            writer.writerow(data_row)




def create_folder_struct(map_type: str, env_type: str, exp_type: str) -> str:
    top_dir = os.path.join(base_dir, map_type)
    mid_dir = os.path.join(top_dir, env_type)
    exp_dir = os.path.join(mid_dir, exp_type)
    return top_dir,mid_dir,exp_dir

def float_to_string(value, decimal_places):
    # Format the float with specified decimal places
    formatted_str = f"{value:.{decimal_places}f}"
    
    # Remove trailing zeros and a trailing decimal point if necessary
    formatted_str = formatted_str.rstrip('0').rstrip('.') if '.' in formatted_str else formatted_str
    
    return formatted_str

def get_exp_name(heuristics: float, analytical: bool, obs_avoid: bool) -> str:
    exp_name = "obs_avoid" if obs_avoid else "skills"
    
    # if heuristics:
    exp_name += f"_use_heuristic{float_to_string(heuristics,6)}"
    exp_name += "_analytical_cost" if analytical else "_predicted_cost"
    
    # Assuming ground_plan is defined elsewhere; add as a function parameter if needed
    ground_plan = 0
    if ground_plan != 0:
        exp_name += "_obs_avoid_planner"
    
    return exp_name

def get_waypoint_file(exp_folder):
    plan_path = os.path.join(exp_folder, "plan.json")
    return plan_path

def get_img_file(img_name):

    img_path = os.path.join(base_dir,"../image", img_name)
    
    return img_path

def gat_num_goals(planner_config):
    dataset_config = read_config(planner_config)
    heuristics = dataset_config["epsilon"]
    analytical = dataset_config["use_analytical"]
    obs_avoid = dataset_config["obs_avoid_planner"]
    map_type = dataset_config["map_type"]
    env_type = dataset_config["predictor_type"]

    

    exp_name = get_exp_name(heuristics,analytical,obs_avoid)
    top_dir,mid_dir,exp_dir = create_folder_struct(map_type,env_type,exp_name)
    file_path = get_waypoint_file(exp_dir)
    # file_path = os.path.join(planning_project_path, path_file_name)

    with open(file_path, "r") as f:
        content = json.load(f)

    waypoints = []
    for i in range(len(content)):
        waypoints.append([content[i]["x"],content[i]["y"] ])

    # list_content = content.split('\n')
    # waypoints = [list_content[i].split(',') for i in range(len(list_content))]
    waypoints = (np.array(waypoints)).astype(float)/2*0.05

    #  set terrain goal to be the waypoints
    waypoints[:, [0, 1]] = waypoints[:, [1, 0]]
    temp_goal = [waypoints[0][0],waypoints[0][1]]
    goals = waypoints[1:-1]
    return goals.shape[0],waypoints[0],goals[0],exp_dir

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # scandots_x,scandots_y,dataset_config, height_map, patchx, patchy = prepare_env(args.config_path)

    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.0,
                                    "parkour_hurdle": 0.0,
                                    "parkour_flat": 0.0,
                                    "parkour_step": 0.0,
                                    "parkour_gap": 0.0, 
                                    "demo": 0.0, 
                                    "parkour_hurdle_edge": 0.0,
                                    "parkour_step_curated":0.0,
                                    "parkour_wall_edge_curated": 0.0,
                                    "parkour_gap_edge": 0.0,
                                    "parkour_wall_edge": 0.0,
                                    "parkour_flat_stop": 0.0,
                                    "wall_edge_distracted_hurdle": 0.0,
                                    "wall_edge_distracted_gap": 0.0,
                                    "gap_edge_distracted_hurdle": 0.0,

                                    "gap_edge_distracted_gap": 0.0,
                                    "hurdle_edge_distracted_hurdle": 0.0,
                                    "wall_edge_distracted_wall": 0.0,
                                    "hurdle_edge_distracted_gap": 0.0,
                                    "hurdle_edge_distracted_wall": 0.0,
                                    "gap_edge_distracted_wall": 0.0,

                                    "gap_distracted_hurdle": 0.0,
                                    "gap_distracted_gap": 0.0,
                                    "hurdle_distracted_hurdle": 0.0,
                                    "hurdle_distracted_gap": 0.0,
                                    "hurdle_distracfsted_wall": 0.0,
                                    "gap_distracted_wall": 0.0,
                                    
                                    "plot_terrain": 0.0,
                                    "policy_test": 0.0,
                                    "img_creation": 0.0,
                                    "energy_terrain": 0.0,
                                    "node_gen_test": 0.0,
                                    "plot_waypoints":0.2
                                    }
                                    

    planner_config = read_planner_config(args.planner_config)
    # if args.plot_mode != 0:
    #     env_cfg.terrain.num_goals = args.plot_mode
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    num_goals, start, target,exp_dir= gat_num_goals(args.planner_config)
    env_cfg.terrain.num_goals = num_goals
    

    # if args.play_waypoints:
    #     env_cfg.terrain.img_path = planner_config["height_map"]
    # else:
    #     env_cfg.terrain.img_path = dataset_config["img_path"]

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_run_conditions(start, target)
    obs = env.get_observations()
    # env.policy_test = dataset_config["policy_test"]
    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(10*int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos, resetcnt = env.step(actions.detach())
        mask_reset_count = resetcnt >= args.num_envs
        if torch.all(mask_reset_count):
            # import ipdb;ipdb.set_trace()
            env.reset_reset_cnt()
            perform_dict = env.get_perform_dict()
            print("perform_dict is: ", perform_dict, "resetcnt is: ", resetcnt, " terrain goal is: ", env.terrain_goals)
            log_data(perform_dict, "legged_gym_result.csv",exp_dir)

        # import ipdb; ipdb.set_trace()
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
