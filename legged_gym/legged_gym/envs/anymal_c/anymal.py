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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg

import math

class Anymal(LeggedRobot):
    cfg : AnymalCRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.control_tick = torch.zeros(
            self.num_envs, 1, dtype=torch.int,
            device=self.device, requires_grad=False)
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.curriculum_index = 0
        if self.num_privileged_obs is not None:
            self.dof_props = torch.zeros((self.num_dofs, 2), device=self.device,
                                         dtype=torch.float)  # includes dof friction (0) and damping (1) for each environment
        self.sim_time = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

    def _get_phase_state(self):
        self.phase_duration = 0.5
        phase = self.sim_time % self.phase_duration
        phi = phase / self.phase_duration # Normalize to [0, 1]

        # Phase offsets for left and right feet
        theta_LFRH= 0.0  # Left-Front, Right-Hind foot is the reference
        theta_RFLH = 0.5  # Right-Front, Left-Hind is half a cycle offset (180 degrees)

        # Compute expected phase activity using Von Mises distribution
        kappa = self.cfg.domain_rand.kappa  # Concentration parameter for Von Mises distribution

        # Helper function to compute P(A_i < f < B_i)
        def compute_phase_probability(phi, theta):
            phi_values_a = torch.linspace(theta - 0.5, theta + 0.5, steps=101, device=self.device).repeat(self.num_envs,1)
            phi_values_b = torch.linspace(theta, theta + 1, steps=101, device=self.device).repeat(self.num_envs, 1)

            von_mises_a = torch.exp(kappa * (torch.cos(2 * math.pi * (phi_values_a - theta))))
            von_mises_a /= torch.sum(von_mises_a, dim=-1, keepdim=True)  # Normalize
            # Create a mask for phi_values_a < phi for each environment
            mask = phi_values_a < phi
            p_a_less_phi = torch.sum(von_mises_a * mask, dim=-1)

            von_mises_b = torch.exp(kappa * (torch.cos(2 * math.pi * (phi_values_b - (theta + 0.5)))))
            von_mises_b /= torch.sum(von_mises_b, dim=-1, keepdim=True)  # Normalize
            # Create a mask for phi_values_a < phi for each environment
            mask = phi_values_b < phi
            p_b_less_phi = torch.sum(von_mises_b * mask, dim=-1)

            p_active = p_a_less_phi * (1 - p_b_less_phi)  # P(A_i < f < B_i)
            return p_active

            # Compute probabilities for left and right feet
        swing_active_left_front = compute_phase_probability(phi, theta_LFRH)
        swing_active_right_hind = compute_phase_probability(phi, theta_LFRH)
        swing_active_right_front = compute_phase_probability(phi, theta_RFLH)
        swing_active_left_hind = compute_phase_probability(phi, theta_RFLH)

        stance_active_left_front = 1.0 - swing_active_left_front
        stance_active_right_hind = 1.0 - swing_active_right_hind
        stance_active_right_front = 1.0 - swing_active_right_front
        stance_active_left_hind = 1.0 - swing_active_left_hind

        return (swing_active_left_front, swing_active_right_hind, swing_active_right_front, swing_active_left_hind,
                stance_active_left_front, stance_active_right_hind, stance_active_right_front, stance_active_left_hind)

    def _reward_synchronous(self):
        swing_active_left_front, swing_active_right_hind, swing_active_right_front, swing_active_left_hind, stance_active_left_front, stance_active_right_hind, stance_active_right_front, stance_active_left_hind = self._get_phase_state()
        lfrh = 1.0 - torch.abs(swing_active_left_front - swing_active_right_hind)
        rflh = 1.0 - torch.abs(swing_active_right_front - swing_active_left_hind)
        phase_sync_reward = lfrh + rflh
        return phase_sync_reward

    def _reward_phase(self):
        # Get phase activity for LFRH and RFLH
        swing_active_left_front, swing_active_right_hind, swing_active_right_front, swing_active_left_hind, stance_active_left_front, stance_active_right_hind, stance_active_right_front, stance_active_left_hind = self._get_phase_state()

        # Contact forces and speeds for each foot
        left_front_foot_forces = self.contact_forces[:, self.feet_indices[0], 2]
        left_hind_foot_forces = self.contact_forces[:, self.feet_indices[1], 2]
        right_front_foot_forces = self.contact_forces[:, self.feet_indices[2], 2]
        right_hind_foot_forces = self.contact_forces[:, self.feet_indices[3], 2]

        #### if not accurate, use jacobian ####
        left_front_foot_speeds = torch.abs(self.dof_vel[:, 2])
        left_hind_foot_speeds = torch.abs(self.dof_vel[:, 5])
        right_front_foot_speeds = torch.abs(self.dof_vel[:, 8])
        right_hind_foot_speeds = torch.abs(self.dof_vel[:, 11])


        # Rewards for stance and swing phases (LFRH)
        stance_reward_left_front = stance_active_left_front * (-1) * left_front_foot_speeds
        stance_reward_right_hind = stance_active_right_hind * (-1) * right_hind_foot_speeds
        swing_reward_left_front = swing_active_left_front * (-1) * left_front_foot_forces
        swing_reward_right_hind = swing_active_right_hind * (-1) * right_hind_foot_forces

        # Rewards for stance and swing phases (RFLH)
        stance_reward_right_front = stance_active_right_front * (-1) * right_front_foot_speeds
        stance_reward_left_hind = stance_active_left_hind * (-1) * left_hind_foot_speeds
        swing_reward_right_front = swing_active_right_front * (-1) * right_front_foot_forces
        swing_reward_left_hind = swing_active_left_hind * (-1) * left_hind_foot_forces

        # Combine rewards
        total_reward = stance_reward_left_front + stance_reward_right_hind + swing_reward_left_front + swing_reward_right_hind + stance_reward_right_front + stance_reward_left_hind + swing_reward_right_front + swing_reward_left_hind

        return total_reward

        

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # LF-RH 대칭성
        error += self.sqrdexp(
            (self.dof_pos[:, 0] - self.dof_pos[:, 9])  # LF와 RH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 1] - self.dof_pos[:, 10])  # LF와 RH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 2] - self.dof_pos[:, 11])  # LF와 RH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_vel[:, 0] - self.dof_vel[:, 9])  # LF와 RH 속도
            / self.cfg.normalization.obs_scales.dof_vel)
        error += self.sqrdexp(
            (self.dof_vel[:, 1] - self.dof_vel[:, 10])  # LF와 RH 속도
            / self.cfg.normalization.obs_scales.dof_vel)
        error += self.sqrdexp(
            (self.dof_vel[:, 2] - self.dof_vel[:, 11])  # LF와 RH 속도
            / self.cfg.normalization.obs_scales.dof_vel)

        # RF-LH 대칭성
        error += self.sqrdexp(
            (self.dof_pos[:, 3] - self.dof_pos[:, 6])  # RF와 LH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 4] - self.dof_pos[:, 7])  # RF와 LH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 5] - self.dof_pos[:, 8])  # RF와 LH
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_vel[:, 3] - self.dof_vel[:, 6])  # RF와 LH 속도
            / self.cfg.normalization.obs_scales.dof_vel)
        error += self.sqrdexp(
            (self.dof_vel[:, 4] - self.dof_vel[:, 7])  # RF와 LH 속도
            / self.cfg.normalization.obs_scales.dof_vel)
        error += self.sqrdexp(
            (self.dof_vel[:, 5] - self.dof_vel[:, 8])  # RF와 LH 속도
            / self.cfg.normalization.obs_scales.dof_vel)

        return error

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)
