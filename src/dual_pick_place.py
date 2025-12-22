import mujoco
import numpy as np

class DualArmPickAndDrop:
    def __init__(self, model, data, left_planner, right_planner, 
                 l_cube_pos, l_tray_pos, r_cube_pos, r_tray_pos):
        self.model = model
        self.data = data
        self.left_planner = left_planner
        self.right_planner = right_planner

        # Constants
        self.POS_THRESHOLD = 0.01
        self.PREGRASP_HEIGHT = 0.07
        self.POSTGRASP_HEIGHT = 0.20
        self.TRAY_DROP_HEIGHT = 0.10 * 2.2

        # Site IDs
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_left")
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site_right")

        # Define targets for Left Arm
        self.l_targets = {
            "PREGRASP": l_cube_pos + np.array([0, 0, self.PREGRASP_HEIGHT]),
            "GRASP": l_cube_pos,
            "LIFT": l_cube_pos + np.array([0, 0, self.POSTGRASP_HEIGHT]),
            "DROP": l_tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT]),
            "DONE": l_tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT])
        }

        # Define targets for Right Arm
        self.r_targets = {
            "PREGRASP": r_cube_pos + np.array([0, 0, self.PREGRASP_HEIGHT]),
            "GRASP": r_cube_pos,
            "LIFT": r_cube_pos + np.array([0, 0, self.POSTGRASP_HEIGHT]),
            "DROP": r_tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT]),
            "DONE": r_tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT])
        }

        # Independent State Trackers
        self.arm_states = {"LEFT": "PREGRASP", "RIGHT": "PREGRASP"}
        self.t_final = None

    def update(self):
        """
        Call this inside your main simulation loop.
        Returns True when BOTH arms are in the 'DONE' state.
        """
        # Process BOTH arms in parallel
        for side in ["LEFT", "RIGHT"]:
            planner = self.left_planner if side == "LEFT" else self.right_planner
            ee_id   = self.left_ee_id if side == "LEFT" else self.right_ee_id
            g_idx   = 7 if side == "LEFT" else 15
            targets = self.l_targets if side == "LEFT" else self.r_targets
            
            state = self.arm_states[side]
            current_target = targets[state]

            # Orientation Logic
            target_quat = np.array([1, 0, 0, 0]) # Default top-down
            gain = 120.0
            # Calculate and apply movement torque
            tau = planner.reach_pose(target_pos=current_target, target_quat=target_quat, pos_gain=gain)
            self.data.ctrl[:] += tau[:]

            # Gripper Logic (0: Closed, 255: Open)
            if state in ["PREGRASP", "GRASP", "DONE"]:
                self.data.ctrl[g_idx] = 255
            else:
                self.data.ctrl[g_idx] = 0

            # Transition Logic
            ee_pos = self.data.site_xpos[ee_id]
            dist = np.linalg.norm(ee_pos - current_target)
            
            if dist < self.POS_THRESHOLD:
                if state == "PREGRASP": self.arm_states[side] = "GRASP"
                elif state == "GRASP":  self.arm_states[side] = "LIFT"
                elif state == "LIFT":   self.arm_states[side] = "DROP"
                elif state == "DROP":   self.arm_states[side] = "DONE"

        # Check if complete
        if all(s == "DONE" for s in self.arm_states.values()):
            if self.t_final is None:
                self.t_final = self.data.time
            # Return True after a 2-second buffer at the end
            if self.data.time - self.t_final > 1.0:
                return True
        
        return False
    
class SingleArmPickAndDrop:
    def __init__(self, model, data, left_planner, right_planner, 
                 cube_pos, tray_pos, side="LEFT"):
        self.model = model
        self.data = data
        self.side = side.upper()
        
        # Select planner and gripper based on side
        self.planner = left_planner if self.side == "LEFT" else right_planner
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 
                                       f"attachment_site_{self.side.lower()}")
        self.g_idx = 7 if self.side == "LEFT" else 15

        # Constants
        self.POS_THRESHOLD = 0.01
        self.PREGRASP_HEIGHT = 0.07
        self.POSTGRASP_HEIGHT = 0.20
        self.TRAY_DROP_HEIGHT = 0.10 * 2.2

        self.targets = {
            "PREGRASP": cube_pos + np.array([0, 0, self.PREGRASP_HEIGHT]),
            "GRASP": cube_pos,
            "LIFT": cube_pos + np.array([0, 0, self.POSTGRASP_HEIGHT]),
            "DROP": tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT]),
            "DONE": tray_pos + np.array([0, 0, self.TRAY_DROP_HEIGHT])
        }

        self.state = "PREGRASP"
        self.start_time = 0.0

    def update(self):
        target_pos = self.targets[self.state]
        target_quat = np.array([1, 0, 0, 0])
        
        # Movement torque for the ACTIVE arm
        gain = 400.0 if self.state == "LIFT" else 120.0
        tau = self.planner.reach_pose(target_pos=target_pos, target_quat=target_quat, pos_gain=gain)
        self.data.ctrl[:] += tau[:]

        # Gripper logic
        if self.state in ["PREGRASP", "GRASP", "DONE"]:
            self.data.ctrl[self.g_idx] = 255
        else:
            self.data.ctrl[self.g_idx] = 0

        # Transitions
        dist = np.linalg.norm(self.data.site_xpos[self.ee_id] - target_pos)
        if dist < self.POS_THRESHOLD:
            if self.state == "PREGRASP": self.state = "GRASP"
            elif self.state == "GRASP":  self.state = "LIFT"
            elif self.state == "LIFT":   self.state = "DROP"
            elif self.state == "DROP":   
                self.state = "DONE"
                self.start_time = self.data.time

        if self.state == "DONE":
            return (self.data.time - self.start_time) > 1.0
            
        return False