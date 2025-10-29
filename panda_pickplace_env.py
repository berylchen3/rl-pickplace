# Gymnasium environment for Panda pick-and-place in PyBullet (SAC-friendly)
# Notes:
# - English comments as requested.
# - Key fixes:
#   (1) One-shot grasp-attempt bonus on open->close transition + contact.
#   (2) Continuous lift reward proportional to object height above table.
#   (3) Goal-shaping only after lift.
#   (4) Small per-step time penalty and gripper hysteresis/lock to avoid chattering.
#   (5) Progress shaping (distance improvements).

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import os
from typing import Tuple

class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render: bool = False, max_steps: int = 250, seed: int | None = None):
        super().__init__()
        self.render_mode = "human" if render else None
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed if seed is not None else 42)

        # --- PyBullet basics ---
        self._physics_client = None
        self.time_step = 1.0 / 240.0
        self._pbkw = dict(physicsClientId=0)  # will be rewritten after connect()

        # --- Scene configuration ---
        self.table_top_z = 0.0        # will be set after loading table
        self.ee_step_scale = 0.03     # per-step position delta scale (meters)
        self.ee_limits = np.array([   # workspace box [xmin, xmax, ymin, ymax, zmin, zmax]
            [-0.4, 0.4, -0.3, 0.3, 0.02, 0.45]
        ]).reshape(-1, 6)[0]

        # --- Object / goal ---
        self.cube_half = 0.02   # 4 cm edge cube
        self.goal_radius = 0.03 # goal sphere visual radius (contact tolerance uses smaller)

        # --- IDs (filled after load) ---
        self.plane_uid = None
        self.table_uid = None
        self.panda_uid = None
        self.object_uid = None
        self.goal_uid = None

        # --- Gripper state and control ---
        self._gripper_open = 1.0  # 1.0=open, 0.0=closed
        self._prev_gripper_open = 1.0
        self._gave_grasp_bonus = False
        self._close_lock_steps = 0  # lock a few steps after closing
        self._close_lock_len = 6    # how many steps to lock after a close command
        self._open_thr = 0.6        # hysteresis thresholds for grip command
        self._close_thr = 0.4

        # --- Episode bookkeeping ---
        self._step_count = 0
        self._prev_dist_ee_obj = None
        self._prev_dist_obj_goal = None
        self._lifted_once = False

        # --- Action/Observation spaces ---
        # action: [dx, dy, dz, grip_command], each in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # observation: [ee_pos(3), obj_pos(3), goal_pos(3), gripper_open(1), dist_to_obj(1), dist_obj_goal(1)]
        high = np.array([np.inf] * 3 * 3 + [1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if render:
            self._connect(gui=True)
        else:
            self._connect(gui=False)

        self._load_scene()

    # -------------- PyBullet init / teardown --------------
    def _connect(self, gui: bool):
        if gui:
            cid = p.connect(p.GUI)
        else:
            cid = p.connect(p.DIRECT)
        self._pbkw = dict(physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._pbkw)
        p.setGravity(0, 0, -9.81, **self._pbkw)
        p.setTimeStep(self.time_step, **self._pbkw)

    def close(self):
        try:
            p.disconnect(**self._pbkw)
        except Exception:
            pass

    # -------------- Scene setup --------------
    def _load_scene(self):
        # Plane
        self.plane_uid = p.loadURDF("plane.urdf", **self._pbkw)

        # Table (use a box)
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.35, 0.02], **self._pbkw)
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.35, 0.02], rgbaColor=[0.7, 0.7, 0.7, 1], **self._pbkw)
        self.table_top_z = 0.02
        self.table_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.0, 0.0, self.table_top_z],
            **self._pbkw
        )

        # Panda arm (use a simplified kinematic end-effector point + gripper fingers)
        # For simplicity and portability here, we model the EE as a kinematic point we move directly
        # and emulate a parallel gripper with two small boxes that follow the EE.
        self._create_ee_and_gripper()

        # Cube object
        self._spawn_cube()

        # Goal sphere (visual only)
        self._spawn_goal()

        # Camera (optional GUI)
        if self.render_mode == "human":
            self._set_gui_camera()

    def _set_gui_camera(self):
        # Top-down-ish view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-65,
            cameraTargetPosition=[0.0, 0.0, self.table_top_z + 0.05],
            **self._pbkw
        )

    # -------------- Minimalistic arm/gripper proxy --------------
    def _create_ee_and_gripper(self):
        # EE proxy: a small sphere (kinematic)
        ee_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, 1], **self._pbkw)
        self.ee_uid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ee_vis,
                                        baseCollisionShapeIndex=-1,
                                        basePosition=[0.2, 0.0, self.table_top_z + 0.12],
                                        **self._pbkw)
        # Gripper fingers: two small boxes parented by constraint to ee
        finger_half = [0.005, 0.015, 0.005]
        finger_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=finger_half, **self._pbkw)
        finger_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=finger_half, rgbaColor=[0, 0, 0, 1], **self._pbkw)

        self.finger_left = p.createMultiBody(baseMass=0.02, baseCollisionShapeIndex=finger_col,
                                             baseVisualShapeIndex=finger_vis,
                                             basePosition=[0.2, -0.015, self.table_top_z + 0.12],
                                             **self._pbkw)
        self.finger_right = p.createMultiBody(baseMass=0.02, baseCollisionShapeIndex=finger_col,
                                              baseVisualShapeIndex=finger_vis,
                                              basePosition=[0.2, +0.015, self.table_top_z + 0.12],
                                              **self._pbkw)

        # Keep fingers following EE with constraints (we will open/close by adjusting their local Y)
        self.con_left = p.createConstraint(
            parentBodyUniqueId=self.ee_uid, parentLinkIndex=-1,
            childBodyUniqueId=self.finger_left, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0], **self._pbkw
        )
        self.con_right = p.createConstraint(
            parentBodyUniqueId=self.ee_uid, parentLinkIndex=-1,
            childBodyUniqueId=self.finger_right, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0], **self._pbkw
        )

        # Friction for better grasping
        for bid in [self.finger_left, self.finger_right]:
            p.changeDynamics(bid, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01, **self._pbkw)

    def _set_gripper_width(self, open_ratio: float):
        # Map [0,1] -> finger offset (Y)
        open_ratio = float(np.clip(open_ratio, 0.0, 1.0))
        width = 0.03 * open_ratio  # max ~3cm
        # Move fingers symmetrically along local Y
        p.changeConstraint(self.con_left, [0, -width/2.0, 0], maxForce=200, **self._pbkw)
        p.changeConstraint(self.con_right, [0, +width/2.0, 0], maxForce=200, **self._pbkw)

    # -------------- Objects / Goal --------------
    def _spawn_cube(self):
        half = self.cube_half
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half], **self._pbkw)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[1, 0, 0, 1], **self._pbkw)

        x = self._rng.uniform(-0.15, 0.15)
        y = self._rng.uniform(-0.15, 0.15)
        z = self.table_top_z + half + 0.001
        self.object_uid = p.createMultiBody(baseMass=0.08, baseCollisionShapeIndex=col,
                                            baseVisualShapeIndex=vis, basePosition=[x, y, z], **self._pbkw)
        p.changeDynamics(self.object_uid, -1, lateralFriction=0.8, rollingFriction=0.01, spinningFriction=0.01, **self._pbkw)

    def _spawn_goal(self):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.goal_radius, rgbaColor=[0, 1, 0, 0.4], **self._pbkw)
        # Random goal not too close to object
        for _ in range(100):
            gx = self._rng.uniform(-0.2, 0.2)
            gy = self._rng.uniform(-0.2, 0.2)
            gz = self.table_top_z + 0.01
            obj_pos = p.getBasePositionAndOrientation(self.object_uid, **self._pbkw)[0]
            if np.linalg.norm(np.array([gx, gy]) - np.array(obj_pos[:2])) > 0.10:
                break
        self.goal_uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=vis, basePosition=[gx, gy, gz], **self._pbkw)

    # -------------- Gym API --------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        p.resetSimulation(**self._pbkw)
        p.setGravity(0, 0, -9.81, **self._pbkw)
        p.setTimeStep(self.time_step, **self._pbkw)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._pbkw)

        self._load_scene()

        # Reset internal states
        self._gripper_open = 1.0
        self._prev_gripper_open = 1.0
        self._set_gripper_width(self._gripper_open)
        self._gave_grasp_bonus = False
        self._close_lock_steps = 0
        self._step_count = 0
        self._lifted_once = False

        ee = np.array(p.getBasePositionAndOrientation(self.ee_uid, **self._pbkw)[0])
        obj = np.array(p.getBasePositionAndOrientation(self.object_uid, **self._pbkw)[0])
        goal = np.array(p.getBasePositionAndOrientation(self.goal_uid, **self._pbkw)[0])

        self._prev_dist_ee_obj = np.linalg.norm(ee - obj)
        self._prev_dist_obj_goal = np.linalg.norm(obj - goal)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # -------- 1) Apply EE motion --------
        dpos = np.clip(action[:3], -1.0, 1.0) * self.ee_step_scale
        ee_pos = np.array(p.getBasePositionAndOrientation(self.ee_uid, **self._pbkw)[0])
        new_pos = ee_pos + dpos

        # clamp to workspace
        xmin, xmax, ymin, ymax, zmin, zmax = self.ee_limits
        new_pos[0] = float(np.clip(new_pos[0], xmin, xmax))
        new_pos[1] = float(np.clip(new_pos[1], ymin, ymax))
        new_pos[2] = float(np.clip(new_pos[2], zmin, zmax))

        p.resetBasePositionAndOrientation(self.ee_uid, new_pos, [0, 0, 0, 1], **self._pbkw)

        # Keep fingers following
        p.stepSimulation(**self._pbkw)

        # -------- 2) Gripper hysteresis + lock --------
        grip_cmd = float(np.clip(action[3], -1.0, 1.0))
        target_open = self._gripper_open
        if self._close_lock_steps > 0:
            # still locked closed for a few steps
            target_open = 0.0
            self._close_lock_steps -= 1
        else:
            if grip_cmd > self._open_thr:
                target_open = 1.0
            elif grip_cmd < self._close_thr:
                target_open = 0.0
                self._close_lock_steps = self._close_lock_len  # lock after close

        # apply if changed
        if abs(target_open - self._gripper_open) > 1e-6:
            self._gripper_open = target_open
            self._set_gripper_width(self._gripper_open)

        # -------- 3) Read state --------
        ee = np.array(p.getBasePositionAndOrientation(self.ee_uid, **self._pbkw)[0])
        obj = np.array(p.getBasePositionAndOrientation(self.object_uid, **self._pbkw)[0])
        goal = np.array(p.getBasePositionAndOrientation(self.goal_uid, **self._pbkw)[0])

        dist_ee_obj = float(np.linalg.norm(ee - obj))
        dist_obj_goal = float(np.linalg.norm(obj - goal))

        # Contact check: any finger touching the cube
        contacts = p.getContactPoints(self.finger_left, self.object_uid, **self._pbkw) + \
                   p.getContactPoints(self.finger_right, self.object_uid, **self._pbkw)
        has_contact = len(contacts) > 0

        # Determine "lifted" (object height above table + small margin)
        lifted_height = obj[2] - (self.table_top_z + self.cube_half + 0.002)
        lifted = lifted_height > 0.01

        # -------- 4) Rewards (shaping + sparse success) --------
        reward = 0.0

        # (A) Approach progress to object (improvement term)
        approach_progress = self._prev_dist_ee_obj - dist_ee_obj
        reward += 2.0 * approach_progress  # positive if getting closer

        # (B) One-shot grasp attempt bonus on open->close + contact + near
        closing_now = (self._prev_gripper_open >= 0.5 and self._gripper_open < 0.5)
        near_object = dist_ee_obj < 0.05
        if (not self._gave_grasp_bonus) and closing_now and near_object and has_contact:
            reward += 10.0
            self._gave_grasp_bonus = True

        # (C) Continuous lift reward
        lift_reward = 40.0 * float(np.clip(lifted_height, 0.0, 0.08))
        reward += lift_reward

        # (D) Goal shaping only after lifted
        if lifted:
            move_progress = self._prev_dist_obj_goal - dist_obj_goal
            reward += -4.0 * dist_obj_goal + 6.0 * move_progress

        # (E) Placement bonus when object is near goal and low (on table)
        at_goal_xy = np.linalg.norm(obj[:2] - goal[:2]) < 0.04
        on_table = abs(obj[2] - (self.table_top_z + self.cube_half + 0.002)) < 0.01
        if lifted and at_goal_xy and on_table:
            reward += 30.0

        # (F) Success bonus (strict)
        success = at_goal_xy and on_table and self._gripper_open > 0.5
        if success:
            reward += 80.0

        # (G) Small time cost to discourage stalling
        reward += -0.05

        # Action smoothness (optional small penalty)
        reward += -0.001 * float(np.linalg.norm(dpos))

        # -------- 5) Bookkeeping --------
        self._prev_dist_ee_obj = dist_ee_obj
        self._prev_dist_obj_goal = dist_obj_goal
        self._prev_gripper_open = self._gripper_open
        if lifted:
            self._lifted_once = True

        terminated = bool(success)
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "success": success,
            "lifted": lifted,
            "dist_ee_obj": dist_ee_obj,
            "dist_obj_goal": dist_obj_goal,
            "reward_lift": lift_reward
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ee = np.array(p.getBasePositionAndOrientation(self.ee_uid, **self._pbkw)[0], dtype=np.float32)
        obj = np.array(p.getBasePositionAndOrientation(self.object_uid, **self._pbkw)[0], dtype=np.float32)
        goal = np.array(p.getBasePositionAndOrientation(self.goal_uid, **self._pbkw)[0], dtype=np.float32)
        dist_ee_obj = np.linalg.norm(ee - obj).astype(np.float32)
        dist_obj_goal = np.linalg.norm(obj - goal).astype(np.float32)
        obs = np.concatenate([ee, obj, goal, [np.float32(self._gripper_open)], [dist_ee_obj], [dist_obj_goal]], dtype=np.float32)
        return obs

    # -------------- Render --------------
    def render(self):
        # For GUI mode, PyBullet already renders.
        # For rgb_array, we capture a top-down view.
        width, height = 640, 480
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, self.table_top_z + 0.02],
            distance=0.8, yaw=45, pitch=-70, roll=0, upAxisIndex=2, **self._pbkw
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.01, farVal=3.0, **self._pbkw)
        img = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, **self._pbkw)
        return img[2]  # RGB array

