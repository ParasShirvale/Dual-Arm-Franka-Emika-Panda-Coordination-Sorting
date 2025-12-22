import mujoco
import random
import numpy as np
import math

class SceneBuilder:
    """
    Docstring for SceneBuilder
    """

    def __init__(
        self,
        include_red_tray: bool = True,
        include_red_cube: bool = True,
        include_green_tray: bool = True,
        include_green_cube: bool = True,
        include_table: bool = True,
        include_random_site_axes: bool = False,
        site1_pos: list = [0.2, 0., 0.5],
        site2_pos: list = [-0.2, 0.0, 0.5],
        file_path = "D:/Robotics/December/franka_emika_panda/dual_panda_scene.xml",
    ):
        self.include_red_tray = include_red_tray
        self.include_red_cube = include_red_cube
        self.include_green_tray = include_green_tray
        self.include_green_cube = include_green_cube
        self.include_table = include_table
        self.include_random_site_axes = include_random_site_axes
        self.site1_pos = site1_pos
        self.site2_pos = site2_pos
        self.file_path = file_path

    def build_dual_arm_robot_scene(self):
        # Path to your MuJoCo XML model (MJCF or converted URDF file)
        FILE_PATH = self.file_path
        robot = mujoco.MjSpec.from_file(FILE_PATH)

        # Table dimensions and position
        table_pos = [0.0, -0.5, 0.175]
        table_length = 0.45
        table_width = 0.2
        table_thickness = 0.025

        # Constants
        min_separation = 0.075  # Minimum separation between cubes
        MAX_ATTEMPTS = 100

        # Randomise Cube position within the table bounds
        cube_half = 0.02

        def random_cube():
            x = random.uniform(table_pos[0] - table_length + cube_half,
                               table_pos[0] + table_length - cube_half)
            y = random.uniform(table_pos[1] - table_width + cube_half,
                               table_pos[1] + table_width - cube_half)
            z = table_pos[2] + table_thickness + cube_half
            return [x, y, z]
        
        # Tray dimensions (half-sizes used by the MJCF box)
        tray_half_x = 0.025
        tray_half_y = 0.025
        tray_z = table_pos[2] + table_thickness + 0.005

        def random_tray():
            x = random.uniform(table_pos[0] - table_length + tray_half_x,
                               table_pos[0] + table_length - tray_half_x)
            y = random.uniform(table_pos[1] - table_width + tray_half_y,
                               table_pos[1] + table_width - tray_half_y)
            return [x, y, tray_z]

        def is_overlapping(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < min_separation
        
        
        def add_site_with_axes(worldbody, site_name, site_position, site_color=[1, 0, 0, 1], site_radius=0.01):
            """
            Adds a site with a coordinate frame (X, Y, Z axes) at the specified position.
            
            Args:
                worldbody: The worldbody to add the site and axes to.
                site_name (str): The name of the site.
                site_position (list or np.array): The [x, y, z] position of the site in world coordinates.
            """
            # Add the site at the given position
            worldbody.add_site(
                name=site_name,
                pos=site_position,
                size=[site_radius, site_radius, site_radius],
                rgba=site_color
            )
            
            # Parameters for the axes
            axis_length = 0.01  # Length of each axis
            axis_radius = 0.2  # Radius of each axis (thickness)
            
            # # Add X-axis (Red)
            # worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[axis_length, axis_radius,0], pos=site_position, quat=[1, 1, 1, 1], rgba=[1, 0, 0, 1])
            
            # # Add Y-axis (Green)
            # worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[axis_length, axis_radius,0], pos=site_position, quat=[-1, 1, 1, 1], rgba=[0, 1, 0, 1])
            
            # # Add Z-axis (Blue)
            # worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[axis_length, axis_radius,0], pos=site_position, quat=[0, 1, 0, 0], rgba=[0, 0, 1, 1])
        
        
        def sample_positions_for_objects():
            """
            Samples non-overlapping positions for:
            - red cube
            - green cube
            - red tray
            - green tray
            using a single global separation distance
            """
            for _ in range(MAX_ATTEMPTS):
                # Sample all objects
                objects = [
                    random_cube(),   # red cube
                    random_cube(),   # green cube
                    random_tray(),   # red tray
                    random_tray(),   # green tray
                ]

                # Check ALL pairwise overlaps
                valid = True
                for i in range(len(objects)):
                    for j in range(i + 1, len(objects)):
                        if is_overlapping(objects[i], objects[j]):
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    cubes = objects[:2]
                    trays = objects[2:]
                    return cubes, trays

            print("⚠️ Warning: Could not find non-overlapping positions")
            return None, None

        # Example usage
        cubes, trays = sample_positions_for_objects()
        
        # 3 things on one side
        # Except one cube
        # cubes = [[-0.32106674642726507, -0.34760148291640025, 0.21999999999999997], [0.1288107660195263, -0.3808533211756263, 0.21999999999999997]]
        # trays = [[0.27120920325891248, -0.5219683622097229, 0.205], [0.0928473876791149, -0.5247787422670399, 0.205]]
        # Except one Tray
        # cubes = [[-0.32106674642726507, -0.34760148291640025, 0.21999999999999997], [-0.1288107660195263, -0.3808533211756263, 0.21999999999999997]]
        # trays = [[-0.27120920325891248, -0.5219683622097229, 0.205], [0.0928473876791149, -0.5247787422670399, 0.205]]
        

        # Both colors on one side
        # LEFT
        # cubes = [[0.32106674642726507, -0.34760148291640025, 0.21999999999999997], [0.1288107660195263, -0.3808533211756263, 0.21999999999999997]]
        # trays = [[0.27120920325891248, -0.5219683622097229, 0.205], [0.0928473876791149, -0.5247787422670399, 0.205]]
        # # RIGHT
        # cubes = [[0.32106674642726507, -0.34760148291640025, 0.21999999999999997], [0.1288107660195263, -0.3808533211756263, 0.21999999999999997]]
        # trays = [[0.27120920325891248, -0.5219683622097229, 0.205], [0.0928473876791149, -0.5247787422670399, 0.205]]

        # Same color on same side
        # Green Left, Red Right
        # cubes = [[-0.08245498991209693, -0.45923869801302486, 0.21999999999999997], [0.20065155039027238, -0.509384362442892, 0.21999999999999997]]
        # trays = [[-0.28185480645876376, -0.4998958914981626, 0.205], [0.2866667367870055, -0.6257711549992029, 0.205]]
        # Red Left, Green Right
        # cubes = [[0.09357897361238593, -0.4577732640566078, 0.21999999999999997], [-0.28005824975044913, -0.6544141691985075, 0.21999999999999997]]
        # trays = [[0.40435603878885956, -0.4315134021248936, 0.205], [-0.04566089927478939, -0.5482658605612919, 0.205]]

        # Opposite colors on ne side
        # cubes = [[-0.32106674642726507, -0.34760148291640025, 0.21999999999999997], [0.1288107660195263, -0.3808533211756263, 0.21999999999999997]]
        # trays = [[0.27120920325891248, -0.5219683622097229, 0.205], [-0.0928473876791149, -0.5247787422670399, 0.205]]

        
        print("Cubes:", cubes)
        print("Trays:", trays)

        # Build Table
        if self.include_table:
            tbl = robot.worldbody.add_body(name='table', pos=table_pos)
            tbl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[table_length, table_width, table_thickness], rgba=[0.8, 0.6, 0.4, 1])
            leg_sz = [0.025, 0.025, 0.075]
            for dx, dy in [(-1,1), (1,1), (-1,-1), (1,-1)]:
                tbl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=leg_sz, 
                             pos=[dx*(table_length-leg_sz[0]), dy*(table_width-leg_sz[1]), -0.1], rgba=[0.8, 0.6, 0.4, 1])

        if self.include_random_site_axes:
            add_site_with_axes(robot.worldbody, "site1", self.site1_pos)
            add_site_with_axes(robot.worldbody, "site2", self.site2_pos)
            

        # Build Cubes
        for i, (name, col) in enumerate([('green_cube', [0.1, 0.8, 0.1, 1]), ('red_cube', [1, 0, 0, 1])]):
            if (i == 0 and self.include_green_cube) or (i == 1 and self.include_red_cube):
                cb = robot.worldbody.add_body(name=name, pos=cubes[i])
                cb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_half]*3, rgba=col, mass=0.06, friction=[0.1, 0.005, 0.005])
                cb.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f'{name}_free')

        # Build Trays with Walls
        for i, (name, col, wall_col) in enumerate([('green_tray', [0.1, 0.8, 0.1, 1], [0.1, 0.5, 0.1, 1]), 
                                                   ('red_tray', [0.8, 0.1, 0.1, 1], [0.5, 0.1, 0.1, 1])]):
            if (i == 0 and self.include_green_tray) or (i == 1 and self.include_red_tray):
                tr = robot.worldbody.add_body(name=name, pos=trays[i])
                tr.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.005], rgba=col)
                # Add 4 walls using a loop
                wall_specs = [[0, 0.05, 0.015, 0.05, 0.005], [0, -0.05, 0.015, 0.05, 0.005], 
                              [0.05, 0, 0.015, 0.005, 0.05], [-0.05, 0, 0.015, 0.005, 0.05]]
                for wx, wy, wz, sx, sy in wall_specs:
                    tr.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.02], pos=[wx, wy, wz], rgba=wall_col)

        # Finalize Physics State
        model = robot.compile()
        data = mujoco.MjData(model)
        arm_home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
        
        initial_qpos = np.zeros(model.nq)
        initial_qpos[0:9], initial_qpos[9:18] = arm_home, arm_home
        
        # Cube Free-Joints: [x, y, z, qw, qx, qy, qz]
        # Red Cube indices: 18-24 | Green Cube indices: 25-31
        initial_qpos[18:21], initial_qpos[21:25] = cubes[0], [1, 0, 0, 0]
        initial_qpos[25:28], initial_qpos[28:32] = cubes[1], [1, 0, 0, 0]

        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
        
        return model, data, [cubes[0], cubes[1], trays[0], trays[1]]