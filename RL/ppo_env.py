import os
import sys
import numpy as np
import time
import mujoco
import mujoco.viewer as viewer
import gymnasium as gym
from gymnasium import spaces

# Setup paths to ensure imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from RL.scene_builder import SceneBuilder
from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner

class SingleArmRL(gym.Env):
    def __init__(self): 
        super().__init__()

        # ------------------- Scene ------------------- #
        self.builder = SceneBuilder(    
            include_green_cube=True,
            include_green_tray=True,
            include_table=True
        )
        self.model, self.data, self.positions = self.builder.build_dual_arm_robot_scene()

        self.planner = DLSVelocityPlanner(
            self.model, self.data,
            site_name="attachment_site_left"
        )

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE,
            "attachment_site_left"
        )

        # ------------------- Actions ------------------- #
        # 0: Go to Cube, 1: Grasp, 2: Go to Tray
        self.action_space = spaces.Discrete(3)

        # ------------------- Observations ------------------- #
        # Fully discrete: distance to cube, distance to tray, is_holding
        self.num_bins = 20
        self.max_dist = 0.5  # maximum distance to scale distances
        self.observation_space = spaces.MultiDiscrete([self.num_bins, self.num_bins, 2])

        # ------------------- Targets ------------------- #
        self.target_cube = self.positions[0]
        self.target_tray = self.positions[2] + np.array([0, 0, 0.15])

        # ------------------- Episode Info ------------------- #
        self.is_holding = False
        self.step_count = 0
        self.max_steps = 50

    # ------------------- Utils ------------------- #
    def _distance_bin(self, dist):
        bin_idx = int((dist / self.max_dist) * self.num_bins)
        return np.clip(bin_idx, 0, self.num_bins - 1)

    def _get_obs(self):
        dist_to_cube = np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target_cube)
        dist_to_tray = np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target_tray)
        cube_bin = self._distance_bin(dist_to_cube)
        tray_bin = self._distance_bin(dist_to_tray)
        grasp_flag = int(self.is_holding)
        return np.array([cube_bin, tray_bin, grasp_flag], dtype=np.int64)

    # ------------------- Step ------------------- #
    def step(self, action, viewer=None):
        self.step_count += 1
        reward = 0.0
        terminated = False

        # ------------------- Determine Target ------------------- #
        if action == 0:  # go to cube
            target_pos = self.target_cube
            gripper_close = False
        elif action == 1:  # grasp
            target_pos = self.target_cube
            gripper_close = True
        elif action == 2:  # go to tray
            target_pos = self.target_tray
            gripper_close = False
        else:
            raise ValueError("Invalid action")

        # ------------------- Apply DLS velocity step ------------------- #
        # Instead of looping until arrival, just apply a single step
        arrived = False
        while not arrived:
            self.data.ctrl.fill(0)
            for aid in range(self.model.nu):
                joint_dof = self.model.actuator_trnid[aid][0]
                self.data.ctrl[aid] = self.data.qfrc_bias[joint_dof]

            tau = self.planner.reach_pose(
                target_pos=target_pos,
                target_quat=np.array([1, 0, 0, 0]),
                pos_gain=150.0
            )
            self.data.ctrl[:] += tau[:]
            self.data.ctrl[7] = 255 if gripper_close else 0

            mujoco.mj_step(self.model, self.data)

            if viewer:
                viewer.sync()
                time.sleep(0.002)
                    
            if np.linalg.norm(self.data.site_xpos[self.ee_id] - target_pos) < 0.01:
                arrived = True
                for _ in range(10):
                    self.data.ctrl[7], self.data.ctrl[15] = 255, 255


        if action == 2:
            self.data.ctrl[7] = 255  # open left gripper
            self.is_holding = False
            
        # Step a few times to let the gripper physically open
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(0.002)

        # ------------------- Reward ------------------- #
        dist_to_cube = np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target_cube)
        dist_to_tray = np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target_tray)


        reward = -dist_to_cube * (1.5 - int(self.is_holding)) - dist_to_tray * (1 - int(self.is_holding))

        # Optional: extra reward for successful grasp or place
        if action == 1 and not self.is_holding:
            self.is_holding = True
        if action == 2 and not self.is_holding:
            self.is_holding = True
            terminated = True
        if self.step_count >= self.max_steps:
            terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    # ------------------- Reset ------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.is_holding = False
        self.step_count = 0
        
        key_id = self.model.key("home1").id
        self.data.qpos[:] = self.builder.set_initial_qpos(
            self.model,
            [self.positions[0], self.positions[1]]
        )
        self.data.ctrl[:] = self.model.key_ctrl[key_id]
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

class DualArmRL(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_bins=20, max_steps=60):
        super().__init__()

        # ------------------- Scene ------------------- #
        self.builder = SceneBuilder(
            include_green_cube=True,
            include_green_tray=True,
            include_table=True
        )
        self.model, self.data, self.positions = self.builder.build_dual_arm_robot_scene()

        # ------------------- Planners ------------------- #
        self.left_planner = DLSVelocityPlanner(
            self.model, self.data,
            site_name="attachment_site_left"
        )
        self.right_planner = DLSVelocityPlanner(
            self.model, self.data,
            site_name="attachment_site_right"
        )

        # ------------------- EE IDs ------------------- #
        self.left_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE,
            "attachment_site_left"
        )
        self.right_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE,
            "attachment_site_right"
        )

        # ------------------- Action Space ------------------- #
        # 0: go to cube
        # 1: grasp
        # 2: go to tray
        self.action_space = spaces.MultiDiscrete([3, 3])

        # ------------------- Observation Space ------------------- #
        # [left_cube_bin, left_tray_bin, left_grasp,
        #  right_cube_bin, right_tray_bin, right_grasp]
        self.num_bins = num_bins
        self.max_dist = 0.5
        self.observation_space = spaces.MultiDiscrete([
            num_bins, num_bins, 2,
            num_bins, num_bins, 2
        ])

        # ------------------- Targets ------------------- #
        self.left_cube = self.positions[0]
        self.right_cube = self.positions[1]
        self.left_tray = self.positions[2] + np.array([0, 0, 0.15])
        self.right_tray = self.positions[3] + np.array([0, 0, 0.15])

        # ------------------- Episode State ------------------- #
        self.left_holding = False
        self.right_holding = False
        self.step_count = 0
        self.max_steps = max_steps

    # ========================================================= #
    #                       UTILITIES                           #
    # ========================================================= #

    def _bin(self, dist):
        idx = int((dist / self.max_dist) * self.num_bins)
        return np.clip(idx, 0, self.num_bins - 1)

    def _get_obs(self):
        left_pos = self.data.site_xpos[self.left_ee]
        right_pos = self.data.site_xpos[self.right_ee]

        return np.array([
            self._bin(np.linalg.norm(left_pos - self.left_cube)),
            self._bin(np.linalg.norm(left_pos - self.left_tray)),
            int(self.left_holding),

            self._bin(np.linalg.norm(right_pos - self.right_cube)),
            self._bin(np.linalg.norm(right_pos - self.right_tray)),
            int(self.right_holding),
        ], dtype=np.int64)

    def _decode_action(self, arm, action):
        if arm == "left":
            cube = self.left_cube
            tray = self.left_tray
        else:
            cube = self.right_cube
            tray = self.right_tray

        if action == 0:
            return cube, False
        elif action == 1:
            return cube, False
        elif action == 2:
            return tray, True
        else:
            raise ValueError("Invalid action")

    # ========================================================= #
    #                          STEP                             #
    # ========================================================= #

    def step(self, action, viewer=None):
        self.step_count += 1
        terminated = False

        left_action, right_action = action

        left_target, left_grip = self._decode_action("left", left_action)
        right_target, right_grip = self._decode_action("right", right_action)

        # ------------------- Apply ONE physics step ------------------- #
        arrived = False
        while not arrived:
            self.data.ctrl.fill(0)

            # gravity compensation
            for aid in range(self.model.nu):
                joint_dof = self.model.actuator_trnid[aid][0]
                self.data.ctrl[aid] = self.data.qfrc_bias[joint_dof]

            tau_l = self.left_planner.reach_pose(
                target_pos=left_target,
                target_quat=np.array([1, 0, 0, 0]),
                pos_gain=150.0
            )

            tau_r = self.right_planner.reach_pose(
                target_pos=right_target,
                target_quat=np.array([1, 0, 0, 0]),
                pos_gain=150.0
            )

            self.data.ctrl[:] += tau_l[:] + tau_r[:]

            # grippers
            self.data.ctrl[7] = 0 if left_grip else 255
            self.data.ctrl[15] = 0 if right_grip else 255

            mujoco.mj_step(self.model, self.data)

            if viewer:
                viewer.sync()
                time.sleep(0.002)
                
            left_check = np.linalg.norm(self.data.site_xpos[self.left_ee] - left_target) < 0.01
            right_check = np.linalg.norm(self.data.site_xpos[self.right_ee] - right_target) < 0.01
            if left_check and right_check:
                arrived = True
                for _ in range(10):
                    self.data.ctrl[7], self.data.ctrl[15] = 255, 255


        if left_action == 2:
            self.data.ctrl[7] = 255  # open left gripper
            self.left_holding = False
        if right_action == 2:
            self.data.ctrl[15] = 255  # open right gripper
            self.right_holding = False
            
        # Step a few times to let the gripper physically open
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(0.002)
                
        # ------------------- Update grasp flags ------------------- #
        if left_action == 1 and not self.left_holding:
            self.left_holding = True

        if right_action == 1 and not self.right_holding:
            self.right_holding = True

        # ------------------- Reward ------------------- #
        reward = 0.0

        left_pos = self.data.site_xpos[self.left_ee]
        right_pos = self.data.site_xpos[self.right_ee]

        reward = np.linalg.norm(left_pos - self.left_cube) * (1 - int(self.left_holding))
        reward -= np.linalg.norm(left_pos - self.left_tray) * int(self.left_holding)

        reward = np.linalg.norm(right_pos - self.right_cube) * (1 - int(self.right_holding))
        reward -= np.linalg.norm(right_pos - self.right_tray) * int(self.right_holding)

        # ------------------- Success ------------------- #
        if (
            left_action == 2 and right_action == 2
            and not self.left_holding and not self.right_holding
        ):
            terminated = True
            self.left_holding = False
            self.right_holding = False

        if self.step_count >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    # ========================================================= #
    #                          RESET                            #
    # ========================================================= #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.left_holding = False
        self.right_holding = False
        self.step_count = 0

        key_id = self.model.key("home1").id
        self.data.qpos[:] = self.builder.set_initial_qpos(
            self.model,
            [self.positions[0], self.positions[1]]
        )
        self.data.ctrl[:] = self.model.key_ctrl[key_id]
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}
    
class Handover(gym.Env):
    def __init__(self):
        super().__init__()

        # ------------------- Scene & Planners ------------------- #
        self.builder = SceneBuilder(include_green_cube=True, include_green_tray=True, include_table=True)
        self.model, self.data, self.positions = self.builder.build_dual_arm_robot_scene()

        self.left_planner = DLSVelocityPlanner(self.model, self.data, site_name="attachment_site_left")
        self.right_planner = DLSVelocityPlanner(self.model, self.data, site_name="attachment_site_right")

        self.left_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_left")
        self.right_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_right")

        # ------------------- Action & Obs Space ------------------- #
        # 0: Pregrasp, 1: Grasp, 2: Handover, 3: Drop (for each arm)
        self.action_space = spaces.MultiDiscrete([4, 4])

        # Obs: [L_dist_cube, L_dist_ho, R_dist_ho, R_dist_tray, L_hold, R_hold, ho_done]
        self.num_bins = 20
        self.observation_space = spaces.MultiDiscrete([self.num_bins, self.num_bins, self.num_bins, self.num_bins, 2, 2, 2])

        # ------------------- Targets ------------------- #
        self.cube_pos = self.positions[0]
        self.tray_pos = self.positions[2] + np.array([0, 0, 0.15])
        
        # Handover Poses (Left=Giver, Right=Receiver)
        self.ho_pos_l = np.array([0.0, self.cube_pos[1], 0.35])
        self.ho_pos_r = self.ho_pos_l + np.array([0.01, 0, 0])
        self.ho_quat_l = np.array([1, 1, 1, 1])
        self.ho_quat_r = np.array([-1, 0, 1, 0])

        self.max_steps = 25
        self.reset()

    def _get_obs(self):
        l_pos = self.data.site_xpos[self.left_ee]
        r_pos = self.data.site_xpos[self.right_ee]

        return np.array([
            self._bin(np.linalg.norm(l_pos - self.cube_pos)),
            self._bin(np.linalg.norm(l_pos - self.ho_pos_l)),
            self._bin(np.linalg.norm(r_pos - self.ho_pos_r)),
            self._bin(np.linalg.norm(r_pos - self.tray_pos)),
            int(self.left_holding),
            int(self.right_holding),
            int(self.handover_done)
        ], dtype=np.int64)

    def _bin(self, d):
        return np.clip(int((d / 0.5) * self.num_bins), 0, self.num_bins - 1)

    def _decode_arm_action(self, arm, action):
        """Returns (target_pos, target_quat, gripper_closed)"""
        if arm == "left":
            if action == 0: return self.cube_pos + np.array([0, 0, 0.15]), np.array([1, 0, 0, 0]), False
            if action == 1: return self.cube_pos, np.array([1, 0, 0, 0]), True
            if action == 2: return self.ho_pos_l, self.ho_quat_l, False
            if action == 3: return self.cube_pos + np.array([0, 0, 0.15]), self.ho_quat_l, True # Drop at HO
        else: # Right arm
            if action == 0: return self.tray_pos + np.array([0, 0, 0.15]), np.array([1, 0, 0, 0]), False
            if action == 1: return self.tray_pos, np.array([1, 0, 0, 0]), True # Reach for HO
            if action == 2: return self.ho_pos_r, self.ho_quat_r, True  # Grasp at HO
            if action == 3: return self.tray_pos, np.array([1, 0, 0, 0]), False

    def step(self, action, viewer=None):
        self.step_count += 1
        l_act, r_act = action
        terminated = False
        
        l_target, l_quat, l_grip = self._decode_arm_action("left", l_act)
        r_target, r_quat, r_grip = self._decode_arm_action("right", r_act)

        # --- Physics Loop (Move until both arrive) ---
        arrived = False
        while not arrived:
            self.data.ctrl.fill(0)
            # Gravity compensation
            for i in range(self.model.nu):
                self.data.ctrl[i] = self.data.qfrc_bias[self.model.actuator_trnid[i][0]]

            tau_l = self.left_planner.reach_pose(l_target, l_quat, pos_gain=150.0)
            tau_r = self.right_planner.reach_pose(r_target, r_quat, pos_gain=150.0)
            
            self.data.ctrl[:] += tau_l + tau_r
            self.data.ctrl[7] = 255 if l_grip else 0
            self.data.ctrl[15] = 255 if r_grip else 0

            mujoco.mj_step(self.model, self.data)
            if viewer: viewer.sync()

            l_dist = np.linalg.norm(self.data.site_xpos[self.left_ee] - l_target)
            r_dist = np.linalg.norm(self.data.site_xpos[self.right_ee] - r_target)
            if l_dist < 0.01 and r_dist < 0.01: arrived = True

        # --- Improved Reward Shaping ---
        reward = -0.1  # Constant step penalty to encourage speed

        # 1. Encourage moving to the cube if not holding
        if not self.left_holding:
            dist_l_cube = np.linalg.norm(self.data.site_xpos[self.left_ee] - self.cube_pos)
            reward += 0.5 * (1.0 - np.tanh(5.0 * dist_l_cube)) # Bonus for proximity to cube

        # 2. Grasping Success
        if l_act == 1 and not self.left_holding:
            # if np.linalg.norm(self.data.site_xpos[self.left_ee] - self.cube_pos) < 0.01:
            self.left_holding = True
            reward += 20.0  # Increased reward

        # 3. Encourage moving to Handover ONLY if holding cube
        if self.left_holding and not self.handover_done:
            dist_l_ho = np.linalg.norm(self.data.site_xpos[self.left_ee] - self.ho_pos_l)
            dist_r_ho = np.linalg.norm(self.data.site_xpos[self.right_ee] - self.ho_pos_r)
            reward += 0.5 * (1.0 - np.tanh(5.0 * dist_l_ho))
            reward += 0.5 * (1.0 - np.tanh(5.0 * dist_r_ho))

        # 4. Handover Success
        if not self.handover_done and self.left_holding:
            if l_act == 2 and r_act == 2:
                # ee_dist = np.linalg.norm(self.data.site_xpos[self.left_ee] - self.data.site_xpos[self.right_ee])
                # if ee_dist < 0.05:
                    self.handover_done = True
                    self.right_holding = True
                    self.left_holding = False
                    reward += 40.0

        # 5. Encourage moving to Tray ONLY after handover
        if self.handover_done:
            dist_r_tray = np.linalg.norm(self.data.site_xpos[self.right_ee] - self.tray_pos)
            reward += 1.0 * (1.0 - np.tanh(5.0 * dist_r_tray))

        # 6. Final Placement
        if self.handover_done and r_act == 3:
            # if np.linalg.norm(self.data.site_xpos[self.right_ee] - self.tray_pos) < 0.05:
            reward += 100.0
            terminated = True

        # Sparse-ish Penalties
        reward -= 0.1 # Step penalty
        
        if self.step_count >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.left_holding = False
        self.right_holding = False
        self.handover_done = False
        self.step_count = 0
        self.data.qpos[:] = self.builder.set_initial_qpos(self.model, [self.positions[0], self.positions[1]])
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}