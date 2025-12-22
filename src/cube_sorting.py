import os
import sys
import mujoco
import mujoco.viewer
import numpy as np

# Add project root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from scene_builder import SceneBuilder
from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner
from handover import DualArmHandover
from dual_pick_place import DualArmPickAndDrop, SingleArmPickAndDrop

def compensate_gravity(model, data):
    grav = np.zeros_like(data.ctrl)
    for aid in range(model.nu):
        joint_dof = model.actuator_trnid[aid][0]
        grav[aid] = data.qfrc_bias[joint_dof]
    return grav

def main():
    # 1. Setup Scene
    builder = SceneBuilder(include_red_tray=True, include_red_cube=True,
                        include_green_tray=True, include_green_cube=True, include_table=True, 
                        include_random_site_axes=False)
    model, data, positions = builder.build_dual_arm_robot_scene()

    # positions[0]=gc, [1]=rc, [2]=gt, [3]=rt
    gc, rc, gt, rt = positions[0], positions[1], positions[2], positions[3]

    # 2. Planners
    left_planner = DLSVelocityPlanner(model, data, site_name="attachment_site_left")
    right_planner = DLSVelocityPlanner(model, data, site_name="attachment_site_right")

    # 3. Decision Logic Helpers
    is_red_matched = (rc[0] * rt[0] > 0)    # Red cube and tray on same side
    is_green_matched = (gc[0] * gt[0] > 0)  # Green cube and tray on same side
    all_on_one_side = (rc[0] * gc[0] > 0) and (rt[0] * gt[0] > 0) and (rc[0] * rt[0] > 0)

    task_queue = []

    # --- SCENARIO C: All 4 things on one side ---
    if all_on_one_side:
        print(">>> MODE: Scenario C - All on one side (Two Single-Arm Pick/Place)")
        side = "RIGHT" if rc[0] > 0 else "LEFT"
        # Task 1: Green single arm
        task_queue.append(SingleArmPickAndDrop(model, data, left_planner, right_planner, gc, gt, side=side))
        # Task 2: Red single arm
        task_queue.append(SingleArmPickAndDrop(model, data, left_planner, right_planner, rc, rt, side=side))

    # --- SCENARIO A & B: Mixed Matched/Unmatched ---
    elif is_red_matched != is_green_matched:
        print(">>> MODE: Scenario A/B - One Matched, One Handover")
        if is_red_matched:
            # Red is local, Green needs handover
            side_red = "RIGHT" if rc[0] > 0 else "LEFT"
            start_arm_green = "RIGHT" if gc[0] > 0 else "LEFT"
            task_queue.append(SingleArmPickAndDrop(model, data, left_planner, right_planner, rc, rt, side=side_red))
            task_queue.append(DualArmHandover(model, data, left_planner, right_planner, gc, gt, starting_arm=start_arm_green))
        else:
            # Green is local, Red needs handover
            side_green = "RIGHT" if gc[0] > 0 else "LEFT"
            start_arm_red = "RIGHT" if rc[0] > 0 else "LEFT"
            task_queue.append(SingleArmPickAndDrop(model, data, left_planner, right_planner, gc, gt, side=side_green))
            task_queue.append(DualArmHandover(model, data, left_planner, right_planner, rc, rt, starting_arm=start_arm_red))

    # --- ORIGINAL SCENARIOS: Simultaneous or Dual Handover ---
    elif is_red_matched and is_green_matched:
        print(">>> MODE: Simultaneous Pick and Place")
        # Determine left/right assignments
        if rc[0] < 0: l_c, l_t, r_c, r_t = rc, rt, gc, gt
        else:        l_c, l_t, r_c, r_t = gc, gt, rc, rt
        task_queue.append(DualArmPickAndDrop(model, data, left_planner, right_planner, l_c, l_t, r_c, r_t))

    else:
        print(">>> MODE: Sequential Handover (Green -> Red)")
        start_arm_g = "RIGHT" if gc[0] > 0 else "LEFT"
        task_queue.append(DualArmHandover(model, data, left_planner, right_planner, gc, gt, starting_arm=start_arm_g))
        start_arm_r = "RIGHT" if rc[0] > 0 else "LEFT"
        task_queue.append(DualArmHandover(model, data, left_planner, right_planner, rc, rt, starting_arm=start_arm_r))

    # 4. Execution Loop
    current_task_idx = 0
    key_id = model.key("home1").id
    data.ctrl[:] = model.key_ctrl[key_id]
    is_pausing = False
    pause_start_time = 0.0
    PAUSE_DURATION = 1.0

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat = np.array([0, -0.5, 0.4]) 
        viewer.cam.distance = 1.5  # Smaller number = closer zoom
        viewer.cam.elevation = -20 # Angle from the floor
        while viewer.is_running():
            data.ctrl[:] = compensate_gravity(model, data)
            if current_task_idx < len(task_queue):
                if is_pausing:
                    if (data.time - pause_start_time) > PAUSE_DURATION:
                        is_pausing = False
                else:
                    finished = task_queue[current_task_idx].update()
                    if finished:
                        current_task_idx += 1
                        is_pausing = True
                        pause_start_time = data.time
                        
            else:
                print("Pick Place Completed")
                break
            mujoco.mj_step(model, data)
            viewer.sync()
        
        
if __name__ == "__main__":
    for attempts in range(5):
        main()