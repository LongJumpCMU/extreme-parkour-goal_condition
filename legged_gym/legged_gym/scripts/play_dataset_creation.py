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
from legged_gym.utils.math import *
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
import data_set
import csv
from collect_data_set_env import prepare_env, all_valid_pnts, check_directory, valid_waypoint, bounds_valid, on_close_wall_valid, is_wall_between_points, is_distance_safe, is_within_height_limit, read_config, scale_env, pos2idx,pos2idx_array, idx2pos, rotate_point, divide_heading

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def log_data(data, reset_pos, start, end, heading, heading_2, data_file_name):
    # import ipdb; ipdb.set_trace()
    for i in range(len(reset_pos)):
        if os.path.isfile(data_file_name):
            with open(data_file_name, "a", newline='') as f:
                writer = csv.writer(f)
                data_row = [reset_pos[i], start[i], end[i], heading[i].item(), heading_2[i].item(), data['success'][i], data['energy_cost'][i], data['time_cost'][i], data['local_patch'][i]]
                writer.writerow(data_row)
                # f.write(data)
        else:
            with open(data_file_name, "w", newline='') as f:
                writer = csv.writer(f)
                header = ['reset_pos','starting pos',  'ending pos', 'heading', 'delta_heading', 'success', 'energy_cost', 'time_cost', 'local_patch']
                writer.writerow(header)
                data_row = [reset_pos[i], start[i], end[i], heading[i].item(), heading_2[i].item(),data['success'][i], data['energy_cost'][i], data['time_cost'][i], data['local_patch'][i]]
                writer.writerow(data_row)

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 20
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = args.num_regions
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"data_set": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # scandots_x,scandots_y,dataset_config, height_map = prepare_env(args.config_path)
    # env_cfg.terrain.img_path = dataset_config["img_path"]
    # all_start = None
    # all_target = None
    # SCANDOTS_RANGE = [[scandots_x[0], scandots_x[-1]], [scandots_y[0], scandots_y[-1]]]
    # if dataset_config["collect_with_planner"]:
    #     all_start, all_target = all_valid_pnts(scandots_x,scandots_y,SCANDOTS_RANGE,dataset_config, height_map)
    
    depth_latent_buffer = []
    

    #adding terminations
    # save body names from the asset
    # body_names = self.gym.get_asset_rigid_body_names(robot_asset)
    # self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    # self.num_bodies = len(body_names)
    # self.num_dofs = len(self.dof_names)
    # feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

    # prepare environment
    env: LeggedRobot_Dataset

    # print("args.task is: ", args.task)
    if args.web:
        web_viewer.setup(env)

    scandots_x,scandots_y,dataset_config, height_map, patchx, patchy = prepare_env(args.config_path)
    env_cfg.terrain.img_path = dataset_config["img_path"]
    all_start = None
    all_target = None
    SCANDOTS_RANGE = [[patchx[0], patchx[-1]], [patchy[0], patchy[-1]]]
    heading_list = divide_heading(dataset_config["heading_divide"]) #[np.pi/3]
    if not dataset_config["collect_with_planner"]:
        all_valid_pairs = np.array(all_valid_pnts(patchx,patchy,SCANDOTS_RANGE,dataset_config, height_map, heading_list, verbose=True))
    # import ipdb; ipdb.set_trace()
    all_true_start = all_valid_pairs[:,0, 0:2]
    # all_start = all_valid_pairs[:,0, 3:5]
    all_target = all_valid_pairs[:,0, 2:]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, init_reset_pos=all_true_start, init_start=all_start, init_target=all_target, init_heights=height_map[pos2idx_array(all_true_start[:,0], dataset_config), pos2idx_array(all_true_start[:,1], dataset_config)])
    obs = env.get_observations()


    

    termination = False
    start_idx = 0
    next_start = all_valid_pairs[:,start_idx, 0:2]
    # next_target = np.array([all_target[start_idx].tolist()])
    next_target = all_valid_pairs[:,start_idx, 2:]
    env.set_run_conditions(next_start,next_target, dataset_config, height_map[pos2idx_array(all_true_start[:,0], dataset_config), pos2idx_array(all_true_start[:,1], dataset_config)])
    env.set_patch(patchx=patchx, patchy=patchy)
    # import ipdb; ipdb.set_trace()

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

    # for i in range(10*int(env.max_episode_length)):
    
    resetcnt = 0
    while not termination:
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
        mask_reset_count = resetcnt >= args.num_agents
        # will make it such that it waits for all env to finish before proceeding for now, will change later
        
        # if resetcnt >= args.num_envs:
        if torch.all(mask_reset_count):
            perform_dict = env.get_perform_dict()
            print("perform_dict is: ", perform_dict, "resetcnt is: ", resetcnt, " terrain goal is: ", env.terrain_goals)
            # get patch into perform dict
            # import ipdb; ipdb.set_trace()
            abs_yaw =  env.compute_yaw(torch.from_numpy(next_target[:,0:2]), torch.from_numpy(next_start))
            abs_yaw_2 =  env.compute_yaw(torch.from_numpy(next_target[:,2:4]), torch.from_numpy(next_target[:,0:2]))

            # .repeat_interleave(args.num_agents, dim=0)
            np.set_printoptions(threshold=np.inf)
            local_patch = env._get_patch_dataset(next_target[:,0:2], abs_yaw)
            # import ipdb; ipdb.set_trace()
            # need to make an arg to get the name of the csv file
            # see if the csv file exists, if not create a new one, otherwise append to it
            # need to write the bash script to repeatedly run this script with different case
            perform_dict["local_patch"] = (local_patch.cpu()).numpy()
            # import ipdb; ipdb.set_trace()
            data_file_name = args.data_file
            # import ipdb; ipdb.set_trace()
            log_data(perform_dict,next_start.tolist(), next_target[:,0:2].tolist(), next_target[:,2:4].tolist(), wrap_to_pi(abs_yaw), wrap_to_pi(abs_yaw_2-abs_yaw),dataset_config["data_file"])
            env.clear_perform_dict()
            env.reset_reset_cnt()
            if not dataset_config["collect_with_planner"]:
                # change such that it terminates when all target points has been reached, while inputting all starting points as regions, also change termination conditions, also need add three point
                #  also need to randomly sample points in patch
                # change patch size
                
                start_idx += 1
                # import ipdb; ipdb.set_trace()
                termination = start_idx >= all_valid_pairs.shape[1]
                if not termination:
                    # next_start = all_start[start_idx]
                    next_start = all_valid_pairs[:,start_idx, 0:2]

                    next_target = all_valid_pairs[:,start_idx, 2:]

                    # start: reset position for all env based on circular rule
                    # target: actual start position as well as desired final target for all envs
                    env.set_run_conditions(next_start,next_target, dataset_config, height_map[pos2idx_array(next_start[:,0], dataset_config), pos2idx_array(next_start[:,1], dataset_config)])

            # if args.collect_with_planner:
            # run planner and get next point
            # else
            # get the next start and end point
            # return if end of starting and target lists
            
            # do env.set_run_conditions() to set the next reset
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        # print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
        #       "cmd vx", env.commands[env.lookat_id, 0].item(),
        #       "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
