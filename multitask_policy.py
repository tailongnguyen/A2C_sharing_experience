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
			save_name,
			share_exp,
			oracle,
			use_laser,
			use_gae,
			noise_argmax,
			timer
			):

		self.oracle = oracle
		self.map_index = map_index
		self.PGNetwork = policies
		self.ZNetwork = oracle_network

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

		returns = [[] for i in range(self.num_task)]
		advantages = [[] for i in range(self.num_task)]

		if not self.use_gae:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					assert len(ep_rewards) == len(ep_states) == len(ep_next_states)
					ep_dones = list(np.zeros_like(ep_rewards))

					if ep_rewards[-1] != 1:
						last_value = current_values[ep_next_states[-1][0], ep_next_states[-1][1], task_idx]
						ep_returns = _discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:
						ep_returns = _discount_with_dones(ep_rewards, ep_dones, self.gamma)

					returns[task_idx] += ep_returns
					ep_values = [current_values[s[0], s[1], task_idx] for s in ep_states]

					# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
			    	# rewards = R + yV(s')
					advantages[task_idx] += list((np.array(ep_returns) - np.array(ep_values)).astype(np.float32))

		else:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					ep_dones = list(np.zeros_like(ep_rewards))

					if ep_rewards[-1] != 1:
						last_value = current_values[ep_next_states[-1][0], ep_next_states[-1][1], task_idx]
						returns[task_idx] += _discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:

						returns[task_idx] += _discount_with_dones(ep_rewards, ep_dones, self.gamma)

					if ep_rewards[-1] == 1:
						ep_dones[-1] = 1
						last_value = 0
					else:
						last_value = current_values[ep_next_states[-1][0], ep_next_states[-1][1], task_idx]

					ep_values = [current_values[s[0], s[1], task_idx] for s in ep_states]
					advantages[task_idx] += _generalized_advantage_estimate(ep_rewards, ep_dones, ep_values, last_value, self.gamma, self.lamb)

					# returns[task_idx] += self._discount_rewards(ep_rewards, ep_next_states, task_idx, current_values)
					# advantages[task_idx] += self._GAE(ep_rewards, ep_states, ep_next_states, task_idx, current_values)
					
				assert len(returns[task_idx]) == len(advantages[task_idx])

		# Gather statistics from samples to do some miscellacious stuffs
		# and fit to Z-training phase.
		state_dict = {}
		count_dict = {}

		for task_idx in range(self.num_task):
			states[task_idx] = list(np.concatenate(states[task_idx]))
			actions[task_idx] = list(np.concatenate(actions[task_idx]))
			redundant_steps[task_idx] = np.mean(redundant_steps[task_idx])

			assert len(states[task_idx]) == len(actions[task_idx]) == len(advantages[task_idx])
			for state, action, gae in zip(states[task_idx], actions[task_idx], advantages[task_idx]):
				state_index = self.env.state_to_index[state[1]][state[0]]

				if state_index not in state_dict:
					state_dict[state_index] = []
					count_dict[state_index] = []
					for tidx in range(self.num_task):
						state_dict[state_index].append([0.0] * self.env.action_size)
						count_dict[state_index].append([0] * self.env.action_size)

				state_dict[state_index][task_idx][action] += gae
				count_dict[state_index][task_idx][action] += 1

		# Because we will eliminate some samples when considering the clipped importance weight,
		# we will made new placeholders to avoid sensitive deletions in current ones.
		mb_states = [[] for _ in range(self.num_task)]
		mb_actions = [[] for _ in range(self.num_task)]
		mb_advantages = [[] for _ in range(self.num_task)]
		mb_returns = [[] for _ in range(self.num_task)]
		mb_logits = [[] for _ in range(self.num_task)]

		share_states = [[] for _ in range(self.num_task)]
		share_actions = [[] for _ in range(self.num_task)]
		share_advantages = [[] for _ in range(self.num_task)]
		share_logits = [[] for _ in range(self.num_task)]

		z_ss, z_as, z_rs = {},{},{}
		
		if self.share_exp:
			assert self.num_task > 1

			# Prepare dictionary to later give share decisions
			share_dict = {}
			task_mean_policy = {}
			
			for task_idx in range(self.num_task):
				for flat_idx, (state, action, gae) in enumerate(zip(states[task_idx], actions[task_idx], advantages[task_idx])):
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

								# for otidx in range(self.num_task):
								# 	if share_info_task[otidx] == '1':
								# 		share_dict[state_index][tidx].append(1)
								# 	else:
								# 		share_dict[state_index][tidx].append(0)

						# Calculate distrubtion of combination sample
						if task_mean_policy.get((state_index, action),-1) == -1:
							task_mean_policy[state_index, action] = []

							for tidx in range(self.num_task):
								mean_policy_action = 0.0
								count = 0.0
								for otidx in range(self.num_task):
									if share_dict[state_index][tidx][otidx] == 1:
										if otidx == tidx or count_dict[state_index][otidx][action] > 0:
											mean_policy_action += (current_policy[state[0], state[1], otidx, 1][action])
											count += 1	

								mean_policy_action /= count
								task_mean_policy[state_index, action].append(mean_policy_action)
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

						# Calculate distrubtion of combination sample
						if task_mean_policy.get((state_index, action),-1) == -1:
							task_mean_policy[state_index, action] = []

							for tidx in range(self.num_task):
								mean_policy_action = 0.0
								count = 0.0
								for otidx in range(self.num_task):
									if share_dict[state_index][tidx][otidx] == 1:
										if otidx == tidx or count_dict[state_index][otidx][action] > 0:
											mean_policy_action += (current_policy[state[0], state[1], otidx, 1][action])
											count += 1	

								mean_policy_action /= count
								task_mean_policy[state_index, action].append(mean_policy_action)

					# Use share_dict to make final batch of data
					for other_task, share in enumerate(share_dict[state_index][task_idx]):

						if other_task == task_idx:
							assert share == 1

						if share == 1:
							importance_weight = current_policy[state[0], state[1], other_task, 1][action] / task_mean_policy[state_index, action][other_task]

							clip_importance_weight = importance_weight
							if clip_importance_weight > 1.2:
								clip_importance_weight = 1.2
							if clip_importance_weight < 0.8:
								clip_importance_weight = 0.8	

							if (importance_weight <= 1.2 and importance_weight >= 0.8) or (clip_importance_weight * gae > importance_weight * gae):

								if other_task == task_idx:
									mb_states[other_task].append(observations[task_idx][flat_idx])
									mb_actions[other_task].append(converted_actions[task_idx][flat_idx])
									mb_returns[other_task].append(returns[task_idx][flat_idx])
									mb_advantages[other_task].append(advantages[task_idx][flat_idx] * importance_weight)
									mb_logits[other_task].append(task_logits[task_idx][flat_idx])

								else:
									share_states[other_task].append(observations[task_idx][flat_idx])
									share_actions[other_task].append(converted_actions[task_idx][flat_idx])
									share_advantages[other_task].append(advantages[task_idx][flat_idx] * importance_weight)
									share_logits[other_task].append(current_policy[state[0], state[1], task_idx, 0])

			if not self.oracle:
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

			return mb_states,\
					 mb_actions,\
					 mb_returns,\
					 mb_advantages,\
					 mb_logits,\
					 rewards,\
					 share_states,\
					 share_actions,\
					 share_advantages,\
					 share_logits,\
					 redundant_steps,\
					 z_ss,\
					 z_as,\
					 z_rs


		# not sharing
		else:
			return observations,\
					 converted_actions,\
					 returns,\
					 advantages,\
					 task_logits,\
					 rewards,\
					 share_states,\
					 share_actions,\
					 share_advantages,\
					 share_logits,\
					 redundant_steps,\
					 z_ss,\
					 z_as,\
					 z_rs
		
		
	def train(self, sess, saver):
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
			mbshare_states,\
			mbshare_actions,\
			mbshare_advantages,\
			mbshare_logits,\
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
				if not self.share_exp:
					assert len(mbshare_states[task_idx]) == len(mbshare_advantages[task_idx]) == len(mbshare_actions[task_idx]) == len(mbshare_logits[task_idx]) == 0

				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx])
				assert len(mbshare_states[task_idx]) == len(mbshare_actions[task_idx]) == len(mbshare_advantages[task_idx]) == len(mbshare_logits[task_idx])

				policy_loss, value_loss, _, _ = self.PGNetwork[task_idx].learn(sess, 
																				actor_states = mb_states[task_idx] + mbshare_states[task_idx],
																				advantages = mb_advantages[task_idx] + mbshare_advantages[task_idx],
																				actions = mb_actions[task_idx] + mbshare_actions[task_idx], 
																				critic_states = mb_states[task_idx],
																				returns = mb_returns[task_idx],
																				task_logits = mb_logits[task_idx] + mbshare_logits[task_idx]
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
			if epoch % self.save_model == 0:
				saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		