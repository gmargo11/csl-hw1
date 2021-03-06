"""
An example of Gym Wrapper
"""
import time

import numpy as np
from airobot import Robot
from airobot.utils.common import ang_in_mpi_ppi
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat
from airobot.utils.common import quat_multiply
from airobot.utils.common import rotvec2quat
from gym import spaces
from gym import Env
import pybullet as p


class PusherEnv(Env):
	def __init__(self, action_repeat=10, render=False):
		self._action_repeat = action_repeat		
		self.robot = Robot('ur5e_stick', pb=True, pb_cfg={'gui': render, 'realtime':False})
		self.ee_ori = [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
		self._action_bound = 1.0
		self._ee_pos_scale = 0.02
		self._ee_ori_scale = np.pi / 36.0
		self._action_high = np.array([self._action_bound] * 2)
		self.action_space = spaces.Box(low=-self._action_high,
									   high=self._action_high,
									   dtype=np.float32)
		
		self.goal = np.array([0.75, -0.3, 1.0])
		self.init = np.array([0.5, 0.1, 1.0])
		self.init_obj = [0.5, 0.0, 1.0]
		self.robot.arm.reset()
		
		ori = euler2quat([0, 0, np.pi / 2])
		self.table_id = self.robot.pb_client.load_urdf('table/table.urdf',
													   [.5, 0, 0.4],
													   ori,
													   scaling=0.9)

		self.obj_id = self.robot.pb_client.load_geom('cylinder', size=[0.08, 0.05], mass=1,
													 base_pos=self.init_obj,
													 rgba=[1, 0, 0, 1])

		self.marker_id = self.robot.pb_client.load_geom('box', size=0.05, mass=1,
													 base_pos=self.goal.tolist(),
													 rgba=[0, 1, 0, 0.4])
		client_id = self.robot.pb_client.get_client_id()
		
		p.setCollisionFilterGroupMask(self.marker_id, -1, 0, 0, physicsClientId=client_id)
		p.setCollisionFilterPair(self.marker_id, self.table_id, -1, -1, 1, physicsClientId=client_id)

		self.reset()
		state_low = np.full(len(self._get_obs()), -float('inf'))
		state_high = np.full(len(self._get_obs()), float('inf'))
		self.observation_space = spaces.Box(state_low,
											state_high,
											dtype=np.float32)


	def reset(self):		
		self.robot.arm.go_home(ignore_physics=True)
		jnt_pos = self.robot.arm.compute_ik(self.init)
		self.robot.arm.set_jpos(jnt_pos, ignore_physics=True)
		
		self.robot.pb_client.reset_body(self.obj_id, self.init_obj)

		self.ref_ee_ori = self.robot.arm.get_ee_pose()[1]
		self.gripper_ori = 0
		self.timestep = 0
		return self._get_obs()

	def step(self, action):
		self.apply_action(action)
		state = self._get_obs()
		self.timestep += 1
		done = (self.timestep >= 200)
		info = dict()
		reward = self.compute_reward_push(state)
		return state, reward, done, info

	def compute_reward_push(self, state):
		gripper_pos = state[0:3]
		obj_pos = state[3:6]

		dist_obj_goal = np.linalg.norm(obj_pos - self.goal)
		dist_grip_obj = np.linalg.norm(obj_pos - gripper_pos)
		init_dist = np.linalg.norm(self.init_obj - self.goal)

		# bonus for proximity of object to goal
		#reward =  5 * (- dist_obj_goal)
		reward = 0

		if dist_obj_goal > 0.05:
			reward = -dist_grip_obj**2 * 40

		# bonus for alignment of pusher, object, goal
		a = (self.goal[1] - obj_pos[1]) 
		b = (self.goal[0] - obj_pos[0])
		d = abs(a * gripper_pos[0] - b * gripper_pos[1] - (a * obj_pos[0] - b * obj_pos[1])) / (a**2 + b**2) # distance from pusher to line
		#print(d)
		is_aligned = ((obj_pos[1] - self.goal[1]) * (gripper_pos[1] - obj_pos[1]) > 0) and ((obj_pos[0] - self.goal[0]) * (gripper_pos[0] - obj_pos[0]) > 0)
		if is_aligned and dist_obj_goal > 0.05:
			reward += (1 - d)
			if d < 0.4:
				reward += (0.4 - dist_grip_obj)

		#print(d, dist_grip_obj, dist_obj_goal)
			#print(reward)
		#reward = 0

		

		# bonus for proximity to goal
		reward += 30 * (init_dist - dist_obj_goal)

		# extra bonus for goal immediate area
		#if dist_obj_goal < 0.10:
		#	reward += 100 + (0.08 - dist_obj_goal) * 200

		# when goal is reached, move gripper away from puck
		if dist_obj_goal < 0.05:
			reward += 100 + dist_grip_obj * 20
			print("goal achieved!")

		#print('dist:', dist_obj_goal)



		return reward


	def _get_obs(self):
		gripper_pos = self.robot.arm.get_ee_pose()[0]
		obj_pos = self.robot.pb_client.get_body_state(self.obj_id)[0]
		state = np.concatenate([gripper_pos, obj_pos])
		return state

	def apply_action(self, action):
		if not isinstance(action, np.ndarray):
			action = np.array(action).flatten()
		if action.size != 2:
			raise ValueError('Action should be [d_x, d_y].')

		action = np.concatenate([action, np.array([0.])])           
		pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
		pos += action[:3] * self._ee_pos_scale

		rot_vec = np.array([0, 0, 1]) * self.gripper_ori
		rot_quat = rotvec2quat(rot_vec)
		ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
		jnt_pos = self.robot.arm.compute_ik(pos, ori=ee_ori)

		for step in range(self._action_repeat):
			self.robot.arm.set_jpos(jnt_pos)
			self.robot.pb_client.stepSimulation()

	def render(self, **kwargs):
		robot_base = self.robot.arm.robot_base_pos
		self.robot.cam.setup_camera(focus_pt=robot_base,
									dist=3,
									yaw=55,
									pitch=-30,
									roll=0)

		rgb, _ = self.robot.cam.get_images(get_rgb=True,
										   get_depth=False)
		return rgb

	def close(self):
		return
		