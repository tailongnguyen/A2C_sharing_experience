import tensorflow as tf      			# Deep Learning library
import numpy as np           			# Handle matrices
import random                			# Handling random number generation
import time                  			# Handling time calculation
import math
import copy
import threading
import os
import sys 

from rollout import Rollout
from env.map import ENV_MAP
from plot_figure import PlotFigure
from collections import deque			# Ordered collection with ends
from env.terrain import Terrain

class MultitaskPolicy(object):

	def __init__(
			self,
			map_index,
			policies,
			writer,
			write_op,
			action_size,
			num_task,
			num_iters,			
			num_episode,
			num_epochs,
			gamma,
			lamb,
			plot_model,
			save_model,
			save_name,
			share_exp,
			share_weight,
			immortal,
			use_laser,
			use_gae,
			noise_argmax,
			timer
			):

		self.map_index = map_index
		self.PGNetwork = policies

		self.writer = writer
		self.write_op = write_op
		self.action_size = action_size
		self.use_gae = use_gae

		self.num_task = num_task
		self.num_iters = num_iters
		self.num_epochs = num_epochs
		self.num_episode =  num_episode

		self.gamma = gamma
		self.lamb = lamb
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model
		self.immortal = immortal

		self.gradients = [[] for i in range(self.num_task)]
		self.batch_eps = [[] for i in range(self.num_task)]

		self.share_exp = share_exp
		self.share_weight = share_weight
		self.noise_argmax = noise_argmax

		self.env = Terrain(self.map_index, use_laser, immortal)

		assert self.num_task <= self.env.num_task

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task, os.path.join('plot', timer))

		self.rollout = Rollout(num_task = self.num_task, 
							num_episode = self.num_episode, 
							num_iters = self.num_iters, 
							map_index = self.map_index, 
							use_laser = use_laser, 
							noise_argmax = self.noise_argmax,
							immortal = immortal)

	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		
		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				if self.noise_argmax:
					p = sess.run(
								self.PGNetwork[task].actor.logits, 
								feed_dict={
									self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
								})
				else:
					p = sess.run(
								self.PGNetwork[task].actor.pi, 
								feed_dict={
									self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
								})
		
				current_policy[x,y,task] = p.ravel().tolist()
						
		
		if (epoch+1) % self.plot_model == 0 or epoch == 0:
			self.plot_figure.plot(current_policy, epoch + 1)
			# self.gradients = [[] for i in range(self.num_task)]
			# self.batch_eps = [[] for i in range(self.num_task)]
						
		return current_policy

	def _prepare_current_values(self, sess, epoch):
		current_values = {}

		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				v = sess.run(
							self.PGNetwork[task].critic.value, 
							feed_dict={
								self.PGNetwork[task].critic.inputs: [self.env.cv_state_onehot[state_index]],
							})
			
				current_values[x,y,task] = v.ravel().tolist()[0]
						
		
		# if (epoch+1) % self.plot_model == 0 or epoch == 0:
		# 	self.plot_figure.plot(current_values, epoch + 1)

		return current_values

	def _process_experience_normalize(self, sess, states, tasks, actions, drewards, current_policy):
		make_holder = lambda x: [[] for i in range(x)]

		batch_ss, batch_as, batch_drs = [make_holder(self.num_task) for i in range(3)]
		share_ss, share_as, share_drs = [make_holder(self.num_task) for i in range(3)]
		samples = {}
		action_samples = {}

		# break trajectories to samples and put to dictionaries:
		# samples[state,task] = returns
		# action_samples = {}

		for i in range(self.num_task):
			for index, state in enumerate(states[i]):
				
				if (state[0], state[1], tasks[i][index]) not in samples:
					samples[state[0], state[1], tasks[i][index]] = []
					action_samples[state[0], state[1], tasks[i][index]] = []
				
				samples[state[0], state[1], tasks[i][index]].append(drewards[i][index])
				action_samples[state[0], state[1], tasks[i][index]].append(actions[i][index])
		
		# get samples from dictionaries and build trainning batch			
		for v in samples.keys():
			state_index = self.env.state_to_index[v[1]][v[0]]

			# normalize discounted rewards
			# if abs(np.std(samples[v]))>1e-3:
			# 	samples[v] = (np.array(samples[v])-np.mean(samples[v]))/np.std(samples[v])
			
			for i, (reward, action) in enumerate(zip(samples[v], action_samples[v])):

				# original samples
				batch_ss[v[2]].append(self.env.cv_state_onehot[state_index])
				batch_as[v[2]].append(self.env.cv_action_onehot[action_samples[v][i]])

				# interpolate sharing samples only interpolate samples in sharing areas 
				if self.share_exp and self.env.MAP[v[1]][v[0]]==2:
					fx = (current_policy[v[0],v[1],1-v[2]][action]+current_policy[v[0],v[1],v[2]][action])/2
					batch_drs[v[2]].append(current_policy[v[0],v[1],v[2]][action] * reward / fx)

					share_ss[1-v[2]].append(self.env.cv_state_onehot[state_index])
					share_as[1-v[2]].append(self.env.cv_action_onehot[action_samples[v][i]])

					share_drs[1-v[2]].append(current_policy[v[0],v[1],1-v[2]][action] * reward / fx)
				
				else:
					batch_drs[v[2]].append(reward)

		return batch_ss, batch_as, batch_drs, share_ss, share_as, share_drs, samples
	


	def _make_batch(self, sess, epoch):


		def _discount_rewards(episode_rewards, gamma):
			discounted_episode_rewards = np.zeros_like(episode_rewards)
			cumulative = 0.0
			for i in reversed(range(len(episode_rewards))):
				cumulative = cumulative * gamma + episode_rewards[i]
				discounted_episode_rewards[i] = cumulative

			return discounted_episode_rewards.tolist()

		def _discount_with_dones(rewards, dones, gamma):
			discounted = []
			r = 0
			# Start from downwards to upwards like Bellman backup operation.
			for reward, done in zip(rewards[::-1], dones[::-1]):
				r = reward + gamma * r * (1. - done)  # fixed off by one bug
				discounted.append(r)
			return discounted[::-1]

		def _generalized_advantage_estimate(rewards, dones, values, last_value, gamma, lamb):
			advantages = np.zeros_like(rewards)
			lastgaelam = 0

	        # From last step to first step
			for t in reversed(range(len(rewards))):
	            # If t == before last step
				if t == len(rewards) - 1:
					# If a state is done, nextnonterminal = 0
					# In fact nextnonterminal allows us to do that logic

					#if done (so nextnonterminal = 0):
					#    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0) 
					# else (not done)
					    #delta = R + gamma * V(st+1)
					nextnonterminal = 1.0 - dones[-1]

					# V(t+1)
					nextvalue = last_value
				else:
					nextnonterminal = 1.0 - dones[t+1]

					nextvalue = values[t+1]

				# Delta = R(t) + gamma * V(t+1) * nextnonterminal  - V(t)
				delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]

				# Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
				advantages[t] = lastgaelam = delta + gamma * lamb * nextnonterminal * lastgaelam

			return list(advantages)

		current_policy = self._prepare_current_policy(sess, epoch)
		current_values = self._prepare_current_values(sess, epoch)

		'''
		states = [
		    task1		[[---episode_1---],...,[---episode_n---]],
		    task2		[[---episode_1---],...,[---episode_n---]],
		   .
		   .
			task_k      [[---episode_1---],...,[---episode_n---]],
		]
		same as actions, tasks, rewards, values, dones
		
		last_values = [
			task1		[---episode_1---, ..., ---episode_n---],
		    task2		[---episode_1---, ..., ---episode_n---],
		   .
		   .
			task_k      [---episode_1---, ..., ---episode_n---],	
		]
		'''
		states, tasks, actions, rewards, values, dones, last_values = self.rollout.rollout_batch(sess, current_policy, current_values, epoch)

		observations = [[] for i in range(self.num_task)]
		converted_actions = [[] for i in range(self.num_task)]
		for task_idx, task_states in enumerate(states):
			for ep_idx, ep_states in enumerate(task_states):
				observations[task_idx] += [self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]]  for s in ep_states]
				converted_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[task_idx][ep_idx]]

		returns = [[] for i in range(self.num_task)]
		advantages = [[] for i in range(self.num_task)]

		if not self.use_gae:

			for task_idx in range(self.num_task):
				for ep_idx, ep_rewards in enumerate(rewards[task_idx]):

					if dones[task_idx][ep_idx][-1] == 0:
						try:
							returns[task_idx] += _discount_with_dones(ep_rewards + [last_values[task_idx][ep_idx]], dones[task_idx][ep_idx]+[0], self.gamma)[:-1]
						except IndexError:
							print("IndexError at MultitaskPolicy!")
							print(states, actions, rewards, values, dones, last_values)
							sys.exit()
					else:
						returns[task_idx] += _discount_with_dones(ep_rewards, dones[task_idx][ep_idx], self.gamma)

				# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
		    	# rewards = R + yV(s')
				advantages[task_idx] = list((np.array(returns[task_idx]) - np.concatenate(values[task_idx])).astype(np.float32))

		else:

			for task_idx in range(self.num_task):
				for ep_idx, ep_rewards in enumerate(rewards[task_idx]):
					advantages[task_idx] += _generalized_advantage_estimate(ep_rewards, dones[task_idx][ep_idx], values[task_idx][ep_idx], last_values[task_idx][ep_idx], self.gamma, self.lamb)

				returns[task_idx] = list((np.array(advantages[task_idx]) + np.concatenate(values[task_idx])).astype(np.float32))

		for task_idx in range(self.num_task):
			rewards[task_idx] = np.concatenate(rewards[task_idx])
			states[task_idx] = np.concatenate(states[task_idx])
			actions[task_idx] = np.concatenate(actions[task_idx])

		if self.share_exp:
			assert self.num_task > 1
			for task_idx in range(self.num_task):
				for other_task in range(self.num_task):
					if other_task == task_idx:
						continue

					for idx, s in enumerate(states[other_task]):
						if self.env.MAP[s[1]][s[0]] == 2:

							act = actions[other_task][idx]	
							important_weight = self.share_weight * current_policy[s[0], s[1], task_idx][act] + (1 - self.share_weight) *current_policy[s[0], s[1], other_task][act]
							important_weight = current_policy[s[0], s[1], task_idx][act] / important_weight
							
							observations[task_idx].append(self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]])
							converted_actions[task_idx].append(self.env.cv_action_onehot[act])

							returns[task_idx].append(returns[other_task][idx] * important_weight)
							advantages[task_idx].append(advantages[other_task][idx] * important_weight)

		return observations, converted_actions, returns, advantages, rewards
		
		
	def train(self, sess, saver):

		for epoch in range(self.num_epochs):
			print('epoch {}'.format(epoch), end = '\r', flush = True)
			sys.stdout.flush()
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states, mb_actions, mb_returns, mb_advantages, rewards = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	

			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			sum_dict = {}
			for task_idx in range(self.num_task):
				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx])

				policy_loss, value_loss, policy_entropy, total_loss = self.PGNetwork[task_idx].learn(sess, 
																										mb_states[task_idx],
																										mb_actions[task_idx],
																										mb_returns[task_idx],
																										mb_advantages[task_idx]
																									)

				sum_dict[self.PGNetwork[task_idx].mean_reward] = np.mean(rewards[task_idx])
				sum_dict[self.PGNetwork[task_idx].tloss_summary] = total_loss
				sum_dict[self.PGNetwork[task_idx].ploss_summary] = policy_loss
				sum_dict[self.PGNetwork[task_idx].vloss_summary] = value_loss
				sum_dict[self.PGNetwork[task_idx].entropy_summary] = policy_entropy				
				sum_dict[self.PGNetwork[task_idx].steps_per_ep] = len(mb_states[task_idx])

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, epoch+1)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			# SAVE MODEL
			#---------------------------------------------------------------------------------------------------------------------#	
			# if epoch % self.save_model == 0:
			# 	saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		