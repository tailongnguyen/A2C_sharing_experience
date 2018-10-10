import tensorflow as tf      			# Deep Learning library
import numpy as np           			# Handle matrices
import random                			# Handling random number generation
import time                  			# Handling time calculation
import math
import copy
import os
import sys 

from network import *
from utils import noise_and_argmax
from common.multiprocessing_env import SubprocVecEnv
from common.terrain import Terrain, make_env
from plot_figure import PlotFigure

class Runner(object):

	def __init__(
			self,
			args,
			writer,
			gamma,
			lamb,
			test_name,
			save_name,
			timer
			):

		tf.reset_default_graph()
		self.env = Terrain(args.map_index, args.use_laser, args.immortal)
		self.PGNetwork = []
	
		for i in range(args.num_task):
			policy_i = A2C(
							name 					= 'A2C_' + str(i),
							state_size 				= self.env.cv_state_onehot.shape[1], 
							action_size				= self.env.action_size,
							entropy_coeff 			= args.ec,
							value_function_coeff 	= args.vc,
							max_gradient_norm		= args.max_gradient_norm,
							alpha 					= args.alpha,
							epsilon					= args.epsilon,
							learning_rate			= args.lr,
							decay 					= args.decay,
							reuse					= bool(args.share_latent)
							)
			
			print("\nInitialized network {}, with {} trainable weights.".format('A2C_' + str(i), len(policy_i.find_trainable_variables('A2C_' + str(i)))))
			self.PGNetwork.append(policy_i)

		self.writer = writer
		tf.summary.scalar(test_name + "/rewards", tf.reduce_mean([policy.mean_reward for policy in self.PGNetwork], 0))
		# tf.summary.scalar(test_name + "/tloss", tf.reduce_mean([policy.tloss_summary for policy in policies], 0))
		# tf.summary.scalar(test_name + "/ploss", tf.reduce_mean([policy.ploss_summary for policy in policies], 0))
		# tf.summary.scalar(test_name + "/vloss", tf.reduce_mean([policy.vloss_summary for policy in policies], 0))
		# tf.summary.scalar(test_name + "/entropy", tf.reduce_mean([policy.entropy_summary for policy in policies], 0))
		# tf.summary.scalar(test_name + "/nsteps", tf.reduce_mean([policy.steps_per_ep for policy in self.PGNetwork], 0))

		self.write_op = tf.summary.merge_all()

		self.use_gae = args.use_gae
		self.num_task = args.num_task
		self.num_steps = args.num_iters
		self.num_epochs = args.num_epochs
		self.plot_model = args.plot_model
		self.save_model = args.save_model
		self.share_exp = args.share_exp
		self.share_weight = args.share_weight
		self.noise_argmax = args.noise_argmax

		self.gamma = gamma
		self.lamb = lamb
		self.save_name = save_name
		self.sharing_decay = 0.9

		self.envs_task = []
		self.current_states = [[] for _ in range(self.num_task)]

		for i in range(self.num_task):
			envs = [make_env(map_index = args.map_index, use_laser = args.use_laser, immortal = args.immortal, task = i) for _ in range(args.num_episode)]
			self.envs_task.append(SubprocVecEnv(envs))
			self.current_states[i] = self.envs_task[-1].reset()

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task, os.path.join('plot', timer))

	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		
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
		
				current_policy[x,y,task, 0] = logit.ravel().tolist()
				current_policy[x,y,task, 1] = p.ravel().tolist()
						
		
		# if (epoch+1) % self.plot_model == 0 or epoch == 0:
			# self.plot_figure.plot(current_policy, epoch + 1)
						
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

	def roll_out(self, current_policy, current_values, task):
		mb_states, mb_actions, mb_rewards, mb_values, mb_dones = [[] for _ in range(5)]

		states = self.current_states[task]

		for i in range(self.num_steps):
			actions = []
			values = []
			for (x, y) in states:
				if self.noise_argmax:
					logit = current_policy[x, y, task, 0]
					act = noise_and_argmax(logit)
				else:
					pi = current_policy[x, y, task, 1]
					act = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))

				actions.append(act)
				values.append(current_values[x, y, task])

			next_states, rewards, dones = self.envs_task[task].step(actions)

			mb_rewards.append(rewards)
			mb_dones.append(dones)
			mb_actions.append(actions)
			mb_states.append(states)
			mb_values.append(values)

			states = next_states

		mb_last_values = [0] * len(dones)
		for i in range(len(dones)):
			if not dones[i]:
				x, y = states[i]
				mb_last_values[i] = current_values[x, y, task]

		mb_states = np.concatenate(mb_states, 1).reshape((len(self.envs_task[task]), -1, 2))
		mb_actions = np.array(mb_actions).T
		mb_rewards = np.array(mb_rewards, dtype = np.float32).T
		mb_values = np.array(mb_values, dtype = np.float32).T
		mb_dones = np.array(mb_dones).T

		# print("States\n", mb_states)
		# print("Actions\n", mb_actions)
		# print("Rewards\n", mb_rewards)
		# print("Dones\n", mb_dones)
		# print("values\n", mb_values)

		return mb_states, mb_actions, mb_rewards, mb_values, mb_dones, mb_last_values

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
					nextnonterminal = 1.0 - dones[t+1]

					nextvalue = values[t+1]

				# Delta = R(t) + gamma * V(t+1) * nextnonterminal  - V(t)
				delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]

				# Advantage = delta + gamma *  (lambda) * nextnonterminal  * lastgaelam
				advantages[t] = lastgaelam = delta + gamma * lamb * nextnonterminal * lastgaelam

			return list(advantages)

		current_policy = self._prepare_current_policy(sess, epoch)
		current_values = self._prepare_current_values(sess, epoch)

		raw_task_states = [[] for i in range(self.num_task)]
		raw_task_actions = [[] for i in range(self.num_task)]
		task_states = [[] for i in range(self.num_task)]
		task_actions = [[] for i in range(self.num_task)]
		task_returns = [[] for i in range(self.num_task)]
		task_advantages = [[] for i in range(self.num_task)]
		rewards_summary = [[] for i in range(self.num_task)]

		for task_idx in range(self.num_task):
			'''
			states = [
			    [[x11, y11], [x12, y12], ...],
			    ...,
			    [[xn1, yn1], [xn2, yn2], ...],
			]
			
			actions = [
				[a11, a12, ...],
				...,
				[an1, an2, ...]
			] same applies for rewards, values, dones

			last_values = [ lv1, lv2, ..., lvn]
		
			'''
			states, actions, rewards, values, dones, last_values = self.roll_out(current_policy, current_values, task_idx)
			raw_task_states[task_idx] = np.concatenate(states, 0)
			raw_task_actions[task_idx]= np.concatenate(actions, 0)

			for ep_idx, ep_states in enumerate(states):
				task_states[task_idx] += [self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]]  for s in ep_states]
				task_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[ep_idx]]

			rewards_summary[task_idx] = np.sum(np.concatenate(rewards)) / rewards.shape[0]
						
			if not self.use_gae:

				for ep_idx, ep_rewards in enumerate(rewards):

					if dones[ep_idx][-1] == 0:
						try:
							task_returns[task_idx] += _discount_with_dones(list(ep_rewards) + [last_values[ep_idx]], list(dones[ep_idx])+[0], self.gamma)[:-1]
							
						except IndexError:
							print("IndexError at MultitaskPolicy!")
							print(states, actions, rewards, values, dones, last_values)
							sys.exit()
					else:
						task_returns[task_idx] += _discount_with_dones(ep_rewards, dones[ep_idx], self.gamma)

				# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
		    	# rewards = R + yV(s')
				task_advantages[task_idx] = list((np.array(task_returns[task_idx]) - np.concatenate(values)).astype(np.float32))

			else:

				for ep_idx, ep_rewards in enumerate(rewards):

					if dones[ep_idx][-1] == 0:
						try:
							task_returns[task_idx] += _discount_with_dones(list(ep_rewards) + [last_values[ep_idx]], list(dones[ep_idx])+[0], self.gamma)[:-1]
							
						except IndexError:
							print("IndexError at MultitaskPolicy!")
							print(states, actions, rewards, values, dones, last_values)
							sys.exit()
					else:
						task_returns[task_idx] += _discount_with_dones(ep_rewards, dones[ep_idx], self.gamma)

					task_advantages[task_idx] += _generalized_advantage_estimate(ep_rewards, dones[ep_idx], values[ep_idx], last_values[ep_idx], self.gamma, self.lamb)

				# task_returns[task_idx] = list((np.array(task_advantages[task_idx]) + np.concatenate(values[task_idx])).astype(np.float32))

		if self.share_exp:
			assert self.num_task > 1
			if epoch > 2000:
				return task_states, task_actions, task_returns, task_advantages, rewards
				
			# rand = random.random()
			# if rand > self.sharing_decay ** np.log(epoch):
			# 	return observations, converted_actions, returns, advantages, rewards

			for task_idx in range(self.num_task):
				for other_task in range(self.num_task):
					if other_task == task_idx:
						continue

					for idx, s in enumerate(raw_task_states[other_task]):
							
						if self.env.MAP[s[1]][s[0]] == 2:

							act = raw_task_actions[other_task][idx]	
							important_weight = self.share_weight * current_policy[s[0], s[1], task_idx, 1][act] + (1 - self.share_weight) * current_policy[s[0], s[1], other_task, 1][act]
							important_weight = current_policy[s[0], s[1], task_idx, 1][act] / important_weight
							
							task_states[task_idx].append(self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]])
							task_actions[task_idx].append(self.env.cv_action_onehot[act])

							task_returns[task_idx].append(task_returns[other_task][idx] * important_weight)
							task_advantages[task_idx].append(task_advantages[other_task][idx] * important_weight)

		return task_states, task_actions, task_returns, task_advantages, rewards_summary
		
		
	def train(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)

		sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		num_steps = 0
		for epoch in range(self.num_epochs):
			print('epoch {}'.format(epoch))
			sys.stdout.flush()
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states, mb_actions, mb_returns, mb_advantages, rewards_summary = self._make_batch(sess, epoch)
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

				sum_dict[self.PGNetwork[task_idx].mean_reward] = rewards_summary[task_idx]
				# sum_dict[self.PGNetwork[task_idx].tloss_summary] = total_loss
				# sum_dict[self.PGNetwork[task_idx].ploss_summary] = policy_loss
				# sum_dict[self.PGNetwork[task_idx].vloss_summary] = value_loss
				# sum_dict[self.PGNetwork[task_idx].entropy_summary] = policy_entropy				
				# sum_dict[self.PGNetwork[task_idx].steps_per_ep] = len(mb_states[task_idx])

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = sess.run(self.write_op, feed_dict = sum_dict)
			num_steps += len(mb_states[0])

			self.writer.add_summary(summary, epoch + 1)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			# SAVE MODEL
			#---------------------------------------------------------------------------------------------------------------------#	
			# if epoch % self.save_model == 0:
			# 	saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		