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

from legged_gym.envs.base.legged_robot_config_dataset import LeggedRobotCfgDataset, LeggedRobotCfgPPODataset

class Go1DatasetCfg( LeggedRobotCfgDataset ):
    class init_state( LeggedRobotCfgDataset.init_state ):
        # pos = [0.0, 0.0, 0.42] # x,y,z [m]
        pos = [0.0, 0.0, 0.42] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    # class init_state_slope( LeggedRobotCfg.init_state ):
    #     pos = [0.56, 0.0, 0.24] # x,y,z [m]
    #     default_joint_angles = { # = target angles [rad] when action = 0.0
    #         'FL_hip_joint': 0.03,   # [rad]
    #         'RL_hip_joint': 0.03,   # [rad]
    #         'FR_hip_joint': -0.03,  # [rad]
    #         'RR_hip_joint': -0.03,   # [rad]

    #         'FL_thigh_joint': 1.0,     # [rad]
    #         'RL_thigh_joint': 1.9,   # [rad]1.8
    #         'FR_thigh_joint': 1.0,     # [rad]
    #         'RR_thigh_joint': 1.9,   # [rad]

    #         'FL_calf_joint': -2.2,   # [rad]
    #         'RL_calf_joint': -0.9,    # [rad]
    #         'FR_calf_joint': -2.2,  # [rad]
    #         'RR_calf_joint': -0.9,    # [rad]
    #     }
        
    class control( LeggedRobotCfgDataset.control ):
        # PD Drive parameters:
        control_type = 'P'
        # control_type = 'actuator_net' # this is added with actuator net from walk-these-ways, cn try switch between pd control and actuator net to see which is better
        # print("actuator net is here!")
        stiffness = {'joint': 40.}  # [N*m/rad]                 # This maybe for Go1!!!!!!!!!!
        damping = {'joint': 1.0}  #{'joint': 0.5}     # [N*m*s/rad]              # This maybe for Go1!!!!!!!!!!
        # stiffness = {'joint': 30.}  # [N*m/rad]
        # damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False #False#True # this is added with actuator net from walk-these-ways, cn try switch between pd control and actuator net to see which is better
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1.pt"

    class asset( LeggedRobotCfgDataset.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_walktheseways_limitstest.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_a1_limitstest.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_go1_limitstest.urdf'
        
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfgDataset.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        # class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            # dof_pos_limits = -10.0

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]

        measured_points_x_dataset = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] #[-1.2, -1.05, -0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y_dataset = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]#[-1.2, -1.05, -0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9,1.05, 1.2]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
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
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 3

class Go1RoughCfgPPOdata( LeggedRobotCfgPPODataset ):
    class algorithm( LeggedRobotCfgPPODataset.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPODataset.runner ):
        run_name = ''
        # experiment_name = 'rough_a1'
        experiment_name = 'dataset_go1'

  
