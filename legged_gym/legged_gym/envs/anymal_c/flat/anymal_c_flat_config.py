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

from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO


class AnymalCFlatCfg(AnymalCRoughCfg):
    class env(AnymalCRoughCfg.env):
        num_observations = 48

    """
    Tensor Shapes in Environment:
    actions: torch.Size([1, 12])
    base_ang_vel: torch.Size([1, 3])
    base_init_state: torch.Size([13])
    base_lin_vel: torch.Size([1, 3])
    base_quat: torch.Size([1, 4])
    commands: torch.Size([1, 4])
    commands_scale: torch.Size([3])
    contact_forces: torch.Size([1, 17, 3])
    d_gains: torch.Size([12])
    default_dof_pos: torch.Size([1, 12])
    dof_pos: torch.Size([1, 12])
    dof_pos_limits: torch.Size([12, 2])
    dof_state: torch.Size([12, 2])
    dof_vel: torch.Size([1, 12])
    dof_vel_limits: torch.Size([12])
    env_origins: torch.Size([1, 3])
    episode_length_buf: torch.Size([1])
    feet_air_time: torch.Size([1, 4])
    feet_indices: torch.Size([4])
    forward_vec: torch.Size([1, 3])
    gravity_vec: torch.Size([1, 3])
    last_actions: torch.Size([1, 12])
    last_contacts: torch.Size([1, 4])
    last_dof_vel: torch.Size([1, 12])
    last_root_vel: torch.Size([1, 6])
    noise_scale_vec: torch.Size([48])
    obs_buf: torch.Size([1, 48])
    p_gains: torch.Size([12])
    penalised_contact_indices: torch.Size([8])
    projected_gravity: torch.Size([1, 3])
    reset_buf: torch.Size([1])
    rew_buf: torch.Size([1])
    root_states: torch.Size([1, 13])
    sea_cell_state: torch.Size([2, 12, 8])
    sea_cell_state_per_env: torch.Size([2, 1, 12, 8])
    sea_hidden_state: torch.Size([2, 12, 8])
    sea_hidden_state_per_env: torch.Size([2, 1, 12, 8])
    sea_input: torch.Size([12, 1, 2])
    termination_contact_indices: torch.Size([1])
    time_out_buf: torch.Size([1])
    torque_limits: torch.Size([12])
    torques: torch.Size([1, 12])
    
    Contact Forces Tensor Content:
    tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0')
         
         dof_list = [   'LF_HAA', 'LF_HFE', 'LF_KFE', 
                        'LH_HAA', 'LH_HFE', 'LH_KFE', 
                        'RF_HAA', 'RF_HFE', 'RF_KFE', 
                        'RH_HAA', 'RH_HFE', 'RH_KFE']
                        
         body_names = [  'base', 'LF_HAA', 'LF_HIP', 'LF_hip_fixed', 
                        'LF_HFE', 'LF_THIGH', 'LF_thigh_fixed', 'LF_KFE', 
                        'LF_SHANK', 'LF_shank_fixed', 'LF_FOOT', 'LH_HAA', 
                        'LH_HIP', 'LH_hip_fixed', 'LH_HFE', 'LH_THIGH', 
                        'LH_thigh_fixed', 'LH_KFE', 'LH_SHANK', 'LH_shank_fixed', 
                        'LH_FOOT', 'RF_HAA', 'RF_HIP', 'RF_hip_fixed', 
                        'RF_HFE', 'RF_THIGH', 'RF_thigh_fixed', 'RF_KFE', 
                        'RF_SHANK', 'RF_shank_fixed', 'RF_FOOT', 'RH_HAA', 
                        'RH_HIP', 'RH_hip_fixed', 'RH_HFE', 'RH_THIGH', 
                        'RH_thigh_fixed', 'RH_KFE', 'RH_SHANK', 'RH_shank_fixed', 
                        'RH_FOOT', 'battery', 'bottom_shell', 'face_front', 
                        'depth_camera_front_camera', 'depth_camera_front_camera_parent', 'depth_camera_front_color_frame', 
                        'depth_camera_front_color_optical_frame', 'depth_camera_front_depth_optical_frame', 'wide_angle_camera_front_camera',
                        'wide_angle_camera_front_camera_parent', 'face_rear', 'depth_camera_rear_camera', 'depth_camera_rear_camera_parent', 
                        'depth_camera_rear_color_frame', 'depth_camera_rear_color_optical_frame', 'depth_camera_rear_depth_optical_frame', 
                        'wide_angle_camera_rear_camera', 'wide_angle_camera_rear_camera_parent', 'handle', 'hatch', 'remote', 'depth_camera_left_camera', 
                        'depth_camera_left_camera_parent', 'depth_camera_left_color_frame', 'depth_camera_left_color_optical_frame', 'depth_camera_left_depth_optical_frame', 
                        'depth_camera_right_camera', 'depth_camera_right_camera_parent', 'depth_camera_right_color_frame', 'depth_camera_right_color_optical_frame', 
                        'depth_camera_right_depth_optical_frame', 'docking_hatch_cover', 'lidar_cage', 'lidar', 'top_shell', 'imu_link'
                            ]



         feet_names = ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']

         
    """

    class terrain(AnymalCRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False



    class control(AnymalCRoughCfg.control ):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"


    class asset(AnymalCRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(AnymalCRoughCfg.rewards):
        max_contact_force = 350.
        only_positive_rewards = False
        tracking_sigma = 1.0

        class scales(AnymalCRoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            stand_still = -2.
            tracking_lin_vel = 1.0


            synchronous = 0.0005
            phase = 0.0005

            joint_regularization = 0.0
            # feet_contact_forces = -0.01

    class commands(AnymalCRoughCfg.commands):
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 4.
        heading_command = False

        class ranges:
            lin_vel_x = [-0.4, -0.0]  # min max [m/s]
            lin_vel_y = [0, 0]  # min max [m/s]
            ang_vel_yaw = [0, 0]  # min max [rad/s]
            heading = [-3.14, 3.14]


    class domain_rand(AnymalCRoughCfg.domain_rand):
        friction_range = [0., 1.5]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        kappa = 3.


class AnymalCFlatCfgPPO(AnymalCRoughCfgPPO):
    class policy(AnymalCRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AnymalCRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_anymal_c'
        load_run = -1
        max_iterations = 300