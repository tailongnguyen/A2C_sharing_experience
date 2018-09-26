import numpy as np           
import random                
import time                  
import math
import threading
import random

from rollout_thread import RolloutThread
from random import randint
from env.sxsy import SXSY

class Rollout(object):
	
	def __init__(
		self,
		num_task,
		num_episode,
		num_iters,
		map_index,
		use_laser,
		noise_argmax,
		immortal):
		
		self.num_episode = num_episode
		self.num_iters = num_iters

		self.map_index = map_index
		self.init_maps = SXSY[self.map_index]
		self.use_laser = use_laser
		self.immortal = immortal
		self.num_task = num_task
		self.noise_argmax = noise_argmax

		self.states, self.tasks, self.actions, self.rewards, self.values, self.dones, self.last_values = [self.holder_factory(self.num_task) for i in range(7)]

	def _rollout_process(self, sess, task, sx, sy, current_policy, current_values, num_iters):
		thread_rollout = RolloutThread(
									sess = sess,
									task = task,
									start_x = sx,
									start_y = sy,
									num_steps = num_iters,
									policy = current_policy,
									value_function = current_values,
									map_index = self.map_index,
									use_laser = self.use_laser,
									noise_argmax = self.noise_argmax,
									immortal = self.immortal)

		ep_states, ep_tasks, ep_actions, ep_rewards, ep_values, ep_dones, last_value = thread_rollout.rollout()
		
		self.states[task].append(ep_states)
		self.tasks[task].append(ep_tasks)
		self.actions[task].append(ep_actions)
		self.rewards[task].append(ep_rewards)
		self.dones[task].append(ep_dones)
		self.values[task].append(ep_values)
		self.last_values[task].append(last_value)

	def holder_factory(self, size):
		return [ [] for i in range(size) ]

	def rollout_batch(self, sess, current_policy, current_values, epoch):
		self.states, self.tasks, self.actions, self.rewards, self.values, self.dones, self.last_values = [self.holder_factory(self.num_task) for i in range(7)]

		train_threads = []
		
		for i in range(self.num_episode):
			[sx, sy] = self.init_maps[epoch % 1000][i]
			for task in range(self.num_task):
				train_threads.append(threading.Thread(target=self._rollout_process, args=(sess, task, sx, sy, current_policy, current_values, self.num_iters, )))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards, self.values, self.dones, self.last_values