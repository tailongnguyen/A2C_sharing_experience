import sys
import numpy as np           		
import random                		
import time                  		
import math
from utils import noise_and_argmax
from env.terrain import Terrain

class RolloutThread(object):
	
	def __init__(
		self,
		sess,
		task,
		start_x,
		start_y,
		num_steps,
		policy,
		value_function,
		map_index,
		use_laser,
		noise_argmax,
		immortal):
	
		self.sess = sess
		self.task = task
		self.start_x = start_x
		self.start_y = start_y
		self.num_steps = num_steps
		self.policy = policy
		self.value_function = value_function
		self.noise_argmax = noise_argmax
		self.env = Terrain(map_index, use_laser, immortal)

	def rollout(self):
		states, tasks, actions, rewards_of_episode, values, dones = [], [], [], [], [], []
		
		self.env.resetgame(self.task, self.start_x, self.start_y)
		state = self.env.player.getposition()

		step = 0	

		while True:
			if step > self.num_steps:
				break

			if self.noise_argmax:
				logits = self.policy[state[0], state[1], self.task]
				action = noise_and_argmax(logits)
			else:
				pi = self.policy[state[0], state[1], self.task]
				action = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))  # select action w.r.t the actions prob

			value = self.value_function[state[0], state[1], self.task]

			reward, done = self.env.player.action(action)
			
			next_state = self.env.player.getposition()
			
			# Store results
			states.append(state)
			values.append(value)
			tasks.append(self.task)
			dones.append(int(done))
			actions.append(action)
			rewards_of_episode.append(reward)

			state = next_state
			
			if done:   
				break

			step += 1

		if not dones[-1]:
			last_value = self.value_function[state[0], state[1], self.task]
		else:
			last_value = 0

		return states, tasks, actions, rewards_of_episode, values, dones, last_value