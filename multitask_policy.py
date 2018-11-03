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
			oracle_network,
			writer,
			write_op,
			num_task,
			num_iters,			
			num_episode,
			num_epochs,
			gamma,
			lamb,
			plot_model,
			save_model,
			save_dir,
			save_name,
			share_exp,
			oracle,
			use_laser,
			use_gae,
			noise_argmax,
			timer,
			sess,
			pretrained = [], 
			pretrained_dir = []
			):

		self.oracle = oracle
		self.map_index = map_index
		self.PGNetwork = policies
		self.ZNetwork = oracle_network

		self.pretrained = pretrained
		self.pretrained_dir = pretrained_dir

		if len(pretrained) > 0:
			assert share_exp
			assert len(pretrained_dir) == len(pretrained)

		for task, task_dir in zip(pretrained, pretrained_dir):
			try:
				self.PGNetwork[task].restore_model(sess, task_dir)
			except:
				print("Error loading weights of task {}".format(task))
				sys.exit()

			print("Loaded pretrained weights of task {}".format(task))

		self.writer = writer
		self.write_op = write_op
		self.use_gae = use_gae

		self.num_task = num_task
		self.num_iters = num_iters
		self.num_epochs = num_epochs
		self.num_episode =  num_episode

		self.gamma = gamma
		self.lamb = lamb

		self.save_name = save_name
		self.save_dir = save_dir
		self.plot_model = plot_model
		self.save_model = save_model

		self.share_exp = share_exp
		self.noise_argmax = noise_argmax

		self.env = Terrain(self.map_index, use_laser)

		assert self.num_task <= self.env.num_task

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task, os.path.join('plot', timer))

		self.rollout = Rollout(num_task = self.num_task, 
							num_episode = self.num_episode, 
							num_iters = self.num_iters, 
							map_index = self.map_index, 
							use_laser = use_laser, 
							noise_argmax = self.noise_argmax,
							)

	def _discount_rewards(self, episode_rewards, episode_nexts, task, current_value):
		discounted_episode_rewards = np.zeros_like(episode_rewards)
		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			next_value = episode_rewards[i] + self.gamma * next_value  
			discounted_episode_rewards[i] = next_value

		return discounted_episode_rewards.tolist()

	def _GAE(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		ep_GAE = np.zeros_like(episode_rewards)
		TD_error = np.zeros_like(episode_rewards)
		lamda=0.96

		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			TD_error[i] = episode_rewards[i]+self.gamma*next_value-current_value[episode_states[i][0],episode_states[i][1], task]
			next_value = current_value[episode_states[i][0],episode_states[i][1], task]

		ep_GAE[len(episode_rewards)-1] = TD_error[len(episode_rewards)-1]
		weight = self.gamma*lamda
		for i in reversed(range(len(episode_rewards)-1)):
			ep_GAE[i] += TD_error[i]+weight*ep_GAE[i+1]

		return ep_GAE.tolist()	

	def _prepare_current(self, sess, epoch):
		current_policy = {}
		current_values = {}
		current_oracle = {}

		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				logit = sess.run(
							self.PGNetwork[task].actor.logits, 
							feed_dict={
								self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
							})
				p = sess.run(
							self.PGNetwork[task].actor.pi, 
							feed_dict={
								self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
							})
		
				current_policy[x, y, task, 0] = logit.ravel().tolist()
				current_policy[x, y, task, 1] = p.ravel().tolist()
				
				v = sess.run(
							self.PGNetwork[task].critic.value, 
							feed_dict={
								self.PGNetwork[task].critic.inputs: [self.env.cv_state_onehot[state_index]],
							})
			
				current_values[x, y, task] = v.ravel().tolist()[0]

				if self.num_task > 1 and self.share_exp and not self.oracle:
					current_oracle[x, y, task, task] = [0.0, 1.0]


		if self.num_task > 1 and self.share_exp and not self.oracle:
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				for i in range (self.num_task-1):
						for j in range(i+1,self.num_task):	
							o = sess.run(
										self.ZNetwork[i,j].oracle, 
										feed_dict={
											self.ZNetwork[i,j].inputs: [self.env.cv_state_onehot[state_index].tolist()]
										})

							current_oracle[x, y, i, j] = o.ravel().tolist()
							boundary = 0.3
							if current_oracle[x, y, i, j][1] > boundary:
								current_oracle[x, y, i, j][1] -= boundary
								current_oracle[x, y, i, j][0] += boundary
							else:
								current_oracle[x, y, i, j] = [1.0, 0.0] 	
							current_oracle[x, y, j, i] = current_oracle[x, y, i, j]

		if (epoch+1) % self.plot_model == 0:
			self.plot_figure.plot(current_policy, epoch + 1)

		return current_policy, current_values, current_oracle

	def _make_batch(self, sess, epoch):

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
					nextnonterminal = 1.0 - dones[t]

					nextvalue = values[t+1]

				# Delta = R(t) + gamma * V(t+1) * nextnonterminal  - V(t)
				delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]

				# Advantage = delta + gamma *  (lambda) * nextnonterminal  * lastgaelam
				advantages[t] = lastgaelam = delta + gamma * lamb * nextnonterminal * lastgaelam

			# advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
			return list(advantages)

		current_policy, current_values, current_oracle = self._prepare_current(sess, epoch)

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
		states, tasks, actions, rewards, next_states, redundant_steps = self.rollout.rollout_batch(current_policy, epoch)

		observations = [[] for i in range(self.num_task)]
		converted_actions = [[] for i in range(self.num_task)]
		task_logits = [[] for i in range(self.num_task)]

		for task_idx, task_states in enumerate(states):
			for ep_idx, ep_states in enumerate(task_states):
				observations[task_idx] += [self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]]  for s in ep_states]
				converted_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[task_idx][ep_idx]]
				task_logits[task_idx] += [current_policy[s[0], s[1], task_idx, 0] for s in ep_states]

		# Gather statistics from samples to do some miscellacious stuffs
		# and fit to Z-training phase.
		count_dict = {}

		for task_idx in range(self.num_task):
			ep_states = list(np.concatenate(states[task_idx]))
			ep_actions = list(np.concatenate(actions[task_idx]))
			redundant_steps[task_idx] = np.mean(redundant_steps[task_idx])

			for state, action in zip(ep_states, ep_actions):
				state_index = self.env.state_to_index[state[1]][state[0]]

				if state_index not in count_dict:
					count_dict[state_index] = []
					for tidx in range(self.num_task):
						count_dict[state_index].append([0] * self.env.action_size)

				count_dict[state_index][task_idx][action] += 1

		# Prepare dictionary to later give share decisions
		share_dict = {}
		task_mean_policy = {}
		for task_idx in range(self.num_task):
			for state in list(np.concatenate(states[task_idx], 0)) + list(np.concatenate(next_states[task_idx], 0)):
				state_index = self.env.state_to_index[state[1]][state[0]]

				if self.oracle:
					# Get share information from oracle map
					if state_index not in share_dict:
						share_dict[state_index] = []
						share_info = bin(self.env.ORACLE[state[1]][state[0]])[2:].zfill(self.num_task ** 2)
						'''
							E.g: share_info: 1111 (2 task)
						'''
						for tidx in range(self.num_task): 
							share_dict[state_index].append([])
							share_info_task = share_info[tidx*self.num_task:(tidx+1)*self.num_task]

							assert len(share_info_task) == self.num_task
							share_dict[state_index][tidx] = [int(c) for c in share_info_task]

				else:

					# Get share information from Z-networks
					share_dict[state_index] = []

					for tidx in range(self.num_task):
						share_dict[state_index].append([])
						for otidx in range(self.num_task):
						 	share_dict[state_index][tidx].append(0)

						share_dict[state_index][tidx][tidx]=1
								
					for tidx in range(self.num_task-1):
						for otidx in range(tidx+1,self.num_task):
							share_action =np.random.choice(range(2), 
									  p= np.array(current_oracle[state[0],state[1],otidx,tidx])/sum(current_oracle[state[0],state[1],otidx,tidx]))

							share_dict[state_index][tidx][otidx] = share_action
							share_dict[state_index][otidx][tidx] = share_action

		returns = [[] for i in range(self.num_task)]
		advantages = [[] for i in range(self.num_task)]

		if not self.use_gae:

			for task_idx in range(self.num_task):

				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					assert len(ep_rewards) == len(ep_states) == len(ep_next_states)
					ep_dones = list(np.zeros_like(ep_rewards))

					if ep_rewards[-1] != 1:
						last_state = ep_next_states[-1]
						last_state_idx = self.env.state_to_index[last_state[1]][last_state[0]]

						last_value = np.mean([current_values[ep_next_states[-1][0], ep_next_states[-1][1], other_task] \
											for other_task in range(self.num_task) \
											if share_dict[last_state_idx][other_task][task_idx] == 1]) 

						ep_returns = _discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:
						ep_returns = _discount_with_dones(ep_rewards, ep_dones, self.gamma)

					returns[task_idx] += ep_returns

					importance_weights = {}
					for s, a in zip(ep_states, actions[task_idx][ep_idx]):
						count = 0
						sum_policy = 0
						for other_task in range(self.num_task):
							if share_dict[self.env.state_to_index[s[1]][s[0]]][other_task][task_idx] == 1:
								count += 1
								sum_policy += current_policy[s[0], s[1], other_task, 1][a]

						assert count >= 1 and sum_policy > 0
						mean_policy = sum_policy / count
						importance_weights[s[0], s[1]] = current_policy[s[0], s[1], task_idx, 1][a] / mean_policy

					ep_values = [np.mean([current_values[s[0], s[1], other_task] \
						    for other_task in range(self.num_task) \
						    if share_dict[self.env.state_to_index[s[1]][s[0]]][other_task][task_idx] == 1]) * importance_weights[s[0], s[1]] \
							for s in ep_states]

					assert len(ep_values) == len(ep_states) == len(ep_returns)

					# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
			    	# rewards = R + yV(s')
					advantages[task_idx] += list((np.array(ep_returns) - np.array(ep_values)).astype(np.float32))

		else:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					ep_dones = list(np.zeros_like(ep_rewards))

					if ep_rewards[-1] != 1:
						last_state = ep_next_states[-1]
						last_state_idx = self.env.state_to_index[last_state[1]][last_state[0]]
						try:
							last_value = np.mean([current_values[ep_next_states[-1][0], ep_next_states[-1][1], other_task] \
												for other_task in range(self.num_task) \
												if share_dict[last_state_idx][other_task][task_idx] == 1])
						except KeyError:
							print(last_state, last_state_idx)
							sys.exit("KeyError")

						returns[task_idx] += _discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:

						returns[task_idx] += _discount_with_dones(ep_rewards, ep_dones, self.gamma)
						ep_dones[-1] = 1						
						last_value = 0
					
					importance_weights = {}
					for s, a in zip(ep_states, actions[task_idx][ep_idx]):
						count = 0
						sum_policy = 0
						for other_task in range(self.num_task):
							if share_dict[self.env.state_to_index[s[1]][s[0]]][other_task][task_idx] == 1:
								count += 1
								sum_policy += current_policy[s[0], s[1], other_task, 1][a]

						assert count >= 1 and sum_policy > 0
						mean_policy = sum_policy / count
						importance_weights[s[0], s[1]] = current_policy[s[0], s[1], task_idx, 1][a] / mean_policy

					ep_values = [np.mean([current_values[s[0], s[1], other_task] \
						    for other_task in range(self.num_task) \
						    if share_dict[self.env.state_to_index[s[1]][s[0]]][other_task][task_idx] == 1]) * importance_weights[s[0], s[1]] \
							for s in ep_states]

					assert len(ep_values) == len(ep_states)

					advantages[task_idx] += _generalized_advantage_estimate(ep_rewards, ep_dones, ep_values, last_value, self.gamma, self.lamb)

					# returns[task_idx] += self._discount_rewards(ep_rewards, ep_next_states, task_idx, current_values)
					# advantages[task_idx] += self._GAE(ep_rewards, ep_states, ep_next_states, task_idx, current_values)
					
				assert len(returns[task_idx]) == len(advantages[task_idx])

		z_ss, z_as, z_rs = {},{},{}
		if not self.oracle:
			state_dict = {}
			for task_idx in range(self.num_task):
				states[task_idx] = list(np.concatenate(states[task_idx]))
				actions[task_idx] = list(np.concatenate(actions[task_idx]))

				assert len(states[task_idx]) == len(actions[task_idx]) == len(advantages[task_idx])
				for state, action, gae in zip(states[task_idx], actions[task_idx], advantages[task_idx]):
					state_index = self.env.state_to_index[state[1]][state[0]]

					if state_index not in state_dict:
						state_dict[state_index] = []
						for tidx in range(self.num_task):
							state_dict[state_index].append([0.0] * self.env.action_size)

					state_dict[state_index][task_idx][action] += gae

			for i in range (self.num_task-1):
				for j in range(i+1, self.num_task):
					z_ss[i,j] = []
					z_as[i,j] = []
					z_rs[i,j] = []

			for v in state_dict.keys():
				
				for i in range (self.num_task):
					if count_dict[v][i][action]>0:
						state_dict[v][i][action] = state_dict[v][i][action] / count_dict[v][i][action]

				for i in range (self.num_task-1):
					for j in range(i+1, self.num_task):
						for action in range(self.env.action_size):
						
							z_reward = 0.0
							#if state_dict[v][0][action]>0 and state_dict[v][1][action]>0:
							if state_dict[v][i][action]*state_dict[v][j][action] > 0:
								z_reward = min(abs(state_dict[v][i][action]), abs(state_dict[v][j][action]))
								z_action = [0,1]
							
							if state_dict[v][i][action]*state_dict[v][j][action] < 0:
								z_reward = min(abs(state_dict[v][i][action]), abs(state_dict[v][j][action]))
								z_action = [1,0]

							if 	sum(count_dict[v][i]) == 0 and sum(count_dict[v][j]) > 0:
								z_reward = 0.001
								z_action = [1,0]
							
							if 	sum(count_dict[v][j]) == 0 and sum(count_dict[v][i]) > 0:
								z_reward = 0.001
								z_action = [1,0]
							
							if z_reward>0.0:
								z_ss[i,j].append(self.env.cv_state_onehot[v].tolist())
								z_as[i,j].append(z_action)
								z_rs[i,j].append(z_reward)

		return observations,\
				 converted_actions,\
				 returns,\
				 advantages,\
				 task_logits,\
				 rewards,\
				 redundant_steps,\
				 z_ss,\
				 z_as,\
				 z_rs
		
		
	def train(self, sess):

		total_samples = {}

		for epoch in range(self.num_epochs):
			# sys.stdout.flush()
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states,\
			mb_actions,\
			mb_returns,\
			mb_advantages,\
			mb_logits,\
			rewards,\
			mb_redundant_steps,\
			z_ss,\
			z_as,\
			z_rs  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	

			print('epoch {}/{}'.format(epoch + 1, self.num_epochs), end = '\r', flush = True)
			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			if self.num_task > 1 and self.share_exp and not self.oracle:
				for i in range (self.num_task-1):
						for j in range(i+1,self.num_task):
							if len(z_ss[i,j]) > 0:
								sess.run([self.ZNetwork[i,j].train_opt], feed_dict={
																	self.ZNetwork[i,j].inputs: z_ss[i,j],
																	self.ZNetwork[i,j].actions: z_as[i,j], 
																	self.ZNetwork[i,j].rewards: z_rs[i,j] 
																	})	
			else:
				assert len(z_ss) == len(z_as) == len(z_rs) == 0

			sum_dict = {}
			for task_idx in range(self.num_task):


				# Let's make some assertions to make sure everything is bug-free

				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx])

				if task_idx not in self.pretrained:
					# We do not train the trained policies :D
					policy_loss, value_loss, _, _ = self.PGNetwork[task_idx].learn(sess, 
																					actor_states = mb_states[task_idx],
																					advantages = mb_advantages[task_idx],
																					actions = mb_actions[task_idx],
																					critic_states = mb_states[task_idx],
																					returns = mb_returns[task_idx],
																					task_logits = mb_logits[task_idx]
																				)

					sum_dict[self.PGNetwork[task_idx].mean_reward] = np.sum(np.concatenate(rewards[task_idx])) / len(rewards[task_idx])
					sum_dict[self.PGNetwork[task_idx].mean_redundant] = mb_redundant_steps[task_idx]
					sum_dict[self.PGNetwork[task_idx].vloss_summary] = value_loss

					# correct_adv = 0
					# for (estimated_adv, true_adv) in zip(mb_advantages[task_idx], true_advantages[task_idx]):
					# 	if (estimated_adv > 0 and true_adv > 0) or (estimated_adv < 0 and true_adv < 0):
					# 		correct_adv += 1


					# sum_dict[self.PGNetwork[task_idx].aloss_summary] =  correct_adv / len(list(np.concatenate(rewards[task_idx])))
					# sum_dict[self.PGNetwork[task_idx].ploss_summary] = policy_loss
					# sum_dict[self.PGNetwork[task_idx].entropy_summary] = policy_entropy				
					# sum_dict[self.PGNetwork[task_idx].steps_per_ep] = len(mb_states[task_idx])

					if task_idx not in total_samples:
						total_samples[task_idx] = 0
						
					total_samples[task_idx] += len(list(np.concatenate(rewards[task_idx])))

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, np.mean(list(total_samples.values())))
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	

		# SAVE MODEL
		#---------------------------------------------------------------------------------------------------------------------#	
		for i in range(self.num_task):
			if i not in self.pretrained:
				self.PGNetwork[i].save_model(sess, self.save_dir)
		#---------------------------------------------------------------------------------------------------------------------#		