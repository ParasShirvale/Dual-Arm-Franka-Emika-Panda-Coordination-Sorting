import mujoco
import numpy as np

class DualArmHandover:
    def __init__(self, model, data, left_planner, right_planner, cube_pos, tray_pos, starting_arm="LEFT"):
        self.model = model
        self.data = data
        self.starting_arm = starting_arm.upper()

        # Define Giver and Receiver based on input
        if self.starting_arm == "LEFT":
            self.giver_planner = left_planner
            self.receiver_planner = right_planner
            self.giver_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_left")
            self.receiver_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_right")
            self.giver_ctrl_idx = 7
            self.receiver_ctrl_idx = 15
            # Handover offset: Giver stays at center, Receiver offsets slightly
            self.handover_offset = np.array([0.01, 0, -0.01])
            self.target_quat_giver = np.array([1, 1, 1, 1])
            self.target_quat_receiver = np.array([-1, 0, 1, 0])
            # Pre-handover approach directions
            self.approach_offset_g = np.array([-0.05, 0, 0])
            self.approach_offset_r = np.array([0.05, 0, 0])
        else:   
            self.giver_planner = right_planner
            self.receiver_planner = left_planner
            self.giver_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_right")
            self.receiver_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_left")
            self.giver_ctrl_idx = 15
            self.receiver_ctrl_idx = 7
            self.handover_offset = np.array([-0.01, 0, -0.01])
            self.target_quat_giver = np.array([-1, 0, 1, 0])
            self.target_quat_receiver = np.array([1, 1, 1, 1])
            # Pre-handover approach directions (flipped)
            self.approach_offset_g = np.array([0.1, 0, 0])
            self.approach_offset_r = np.array([-0.1, 0, 0])

        # Constants
        self.POS_THRESHOLD = 0.01
        self.PREGRASP_HEIGHT = 0.07
        self.POSTGRASP_HEIGHT = 0.20
        self.TRAY_DROP_HEIGHT = 0.10 * 2.2
        self.WAIT_TIME = 1.5
        self.RELEASE_PAUSE = 0.8

        # Target Poses
        self.cube_pos = cube_pos
        self.tray_pos = tray_pos
        
        self.pregrasp_pos = self.cube_pos + np.array([0, 0, self.PREGRASP_HEIGHT])
        
        # Handover Positions
        self.handover_pos_giver = self.cube_pos + np.array([-self.cube_pos[0], 0, self.POSTGRASP_HEIGHT])
        self.handover_pos_receiver = self.handover_pos_giver + self.handover_offset
        
        # Intermediate Approach Positions (0.1m offset in X)
        self.pre_handover_g = self.handover_pos_giver + self.approach_offset_g
        self.pre_handover_r = self.handover_pos_receiver + self.approach_offset_r

        self.tray_drop_pos = self.tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT])
        self.target_quat_pick = np.array([1, 0, 0, 0])

        self.state = "PREGRASP"
        self.start_time = 0

    def update(self):
        giver_ee_curr = self.data.site_xpos[self.giver_ee_id]
        receiver_ee_curr = self.data.site_xpos[self.receiver_ee_id]

        if self.state == "PREGRASP":
            tau = self.giver_planner.reach_pose(target_pos=self.pregrasp_pos, target_quat=self.target_quat_pick)
            self.data.ctrl[:] += tau
            if np.linalg.norm(giver_ee_curr - self.pregrasp_pos) < self.POS_THRESHOLD:
                self.state = "GRASP"

        elif self.state == "GRASP":
            tau = self.giver_planner.reach_pose(target_pos=self.cube_pos, target_quat=self.target_quat_pick)
            self.data.ctrl[:] += tau
            self.data.ctrl[7], self.data.ctrl[15] = 255, 255 
            if np.linalg.norm(giver_ee_curr - self.cube_pos) < self.POS_THRESHOLD:
                self.state = "PRE_HANDOVER"

        elif self.state == "PRE_HANDOVER":
            # Move to the 0.1m offset positions first
            tau_g = self.giver_planner.reach_pose(target_pos=self.pre_handover_g, target_quat=self.target_quat_giver)
            tau_r = self.receiver_planner.reach_pose(target_pos=self.pre_handover_r, target_quat=self.target_quat_receiver)
            self.data.ctrl[:] += (tau_g + tau_r)
            self.data.ctrl[self.giver_ctrl_idx] = 0        # Giver holds cube
            self.data.ctrl[self.receiver_ctrl_idx] = 255  # Receiver stays open
            
            if (np.linalg.norm(giver_ee_curr - self.pre_handover_g) < self.POS_THRESHOLD and 
                np.linalg.norm(receiver_ee_curr - self.pre_handover_r) < self.POS_THRESHOLD):
                self.state = "MOVE_TO_HANDOVER"

        elif self.state == "MOVE_TO_HANDOVER":
            tau_g = self.giver_planner.reach_pose(target_pos=self.handover_pos_giver, target_quat=self.target_quat_giver)
            tau_r = self.receiver_planner.reach_pose(target_pos=self.handover_pos_receiver, target_quat=self.target_quat_receiver)
            self.data.ctrl[:] += (tau_g + tau_r)
            self.data.ctrl[self.giver_ctrl_idx] = 0
            self.data.ctrl[self.receiver_ctrl_idx] = 255
            
            if (np.linalg.norm(giver_ee_curr - self.handover_pos_giver) < self.POS_THRESHOLD and 
                np.linalg.norm(receiver_ee_curr - self.handover_pos_receiver) < self.POS_THRESHOLD):
                self.start_time = self.data.time
                self.state = "EXCHANGE"

        elif self.state == "EXCHANGE":
            tau_g = self.giver_planner.reach_pose(target_pos=self.handover_pos_giver, target_quat=self.target_quat_giver)
            tau_r = self.receiver_planner.reach_pose(target_pos=self.handover_pos_receiver, target_quat=self.target_quat_receiver)
            self.data.ctrl[:] += (tau_g + tau_r)
            self.data.ctrl[7], self.data.ctrl[15] = 0, 0 
            
            if (self.data.time - self.start_time) > self.WAIT_TIME:
                self.start_time = self.data.time
                self.state = "RELEASE_AND_WAIT"

        elif self.state == "RELEASE_AND_WAIT":
            tau_g = self.giver_planner.reach_pose(target_pos=self.handover_pos_giver, target_quat=self.target_quat_giver)
            tau_r = self.receiver_planner.reach_pose(target_pos=self.handover_pos_receiver, target_quat=self.target_quat_receiver)
            self.data.ctrl[:] += (tau_g + tau_r)
            self.data.ctrl[self.giver_ctrl_idx] = 255
            self.data.ctrl[self.receiver_ctrl_idx] = 0
            
            if (self.data.time - self.start_time) > self.RELEASE_PAUSE:
                self.state = "RECEIVER_MOVE_TO_TRAY"

        elif self.state == "RECEIVER_MOVE_TO_TRAY":
            tau_r = self.receiver_planner.reach_pose(target_pos=self.tray_drop_pos, target_quat=self.target_quat_pick)
            tau_g = self.giver_planner.reach_pose(target_pos=self.pregrasp_pos, target_quat=self.target_quat_pick)
            self.data.ctrl[:] += (tau_g + tau_r)
            self.data.ctrl[self.giver_ctrl_idx] = 255
            self.data.ctrl[self.receiver_ctrl_idx] = 0
            
            if np.linalg.norm(receiver_ee_curr - self.tray_drop_pos) < self.POS_THRESHOLD:
                self.state = "DONE"
                self.start_time = self.data.time  # <--- ADD THIS LINE TO RESET THE TIMER
                # print("At tray, opening gripper...")

        elif self.state == "DONE":
            # Maintain position while opening
            tau_r = self.receiver_planner.reach_pose(target_pos=self.tray_drop_pos, target_quat=self.target_quat_pick)
            self.data.ctrl[:] += tau_r
            
            self.data.ctrl[self.receiver_ctrl_idx] = 255  # Open gripper
            
            # Now this will correctly wait for 1 second relative to arrival time
            if (self.data.time - self.start_time) > 1.0:
                return True
            return (self.data.time - self.start_time) > 1.0
            
        return False