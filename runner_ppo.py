import tensorflow as tf      			# Deep Learning library
import numpy as np           			# Handle matrices
import random                			# Handling random number generation
import time                  			# Handling time calculation
import math
import copy
import os
import sys 

from ppo_network import A2C_PPO
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

			policy_i = A2C_PPO(
							name 					= 'A2C_' + str(i),
							state_size 				= self.env.cv_state_onehot.shape[1], 
							action_size				= self.env.action_size,
							entropy_coeff 			= args.ec,
							value_function_coeff 	= args.vc,
							max_gradient_norm		= args.max_gradient_norm,
							alpha 					= args.alpha,
							epsilon					= args.epsilon,
							clip_param				= args.clip_param,
							learning_rate			= args.lr,
							decay 					= args.decay,
							reuse					= bool(args.share_latent)
							)
			
			print("\nInitialized network {}, with {} trainable weights.".format('A2C_' + str(i), len(policy_i.find_trainable_variables('A2C_' + str(i)))))
			self.PGNetwork.append(policy_i)

		self.writer = writer
		tf.summary.scalar(test_name + "/rewards", tf.reduce_mean([policy.mean_reward for policy in self.PGNetwork], 0))
		tf.summary.scalar(test_name + "/redundants", tf.reduce_mean([policy.mean_redundant for policy in self.PGNetwork], 0))

		self.write_op = tf.summary.merge_all()

		self.num_episode = args.num_episode
		self.num_task = args.num_task
		self.num_steps = args.num_iters
		self.num_epochs = args.num_epochs

		self.share_exp = args.share_exp
		self.share_decay = args.share_decay
		self.share_cut = args.share_cut

		self.lamb = lamb
		self.gamma = gamma
		self.use_gae = args.use_gae
		self.noise_argmax = args.noise_argmax

		self.save_name = save_name
		self.plot_model = args.plot_model
		self.save_model = args.save_model

		self.ppo_epochs = args.ppo_epochs
		self.mini_batch_size = args.minibatch

		self.envs_task = []
		self.current_states = [[] for _ in range(self.num_task)]

		for i in range(self.num_task):
			envs = [make_env(map_index = args.map_index, 
							use_laser = args.use_laser, 
							immortal = args.immortal, 
							task = i, 
							save_folder = os.path.join('plot', timer, self.save_name)) for _ in range(args.num_episode)]

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
						
		
		if (epoch+1) % self.plot_model == 0:
			self.plot_figure.plot(current_policy, epoch + 1)
						
		return current_policy

	def _prepare_current_values(self, sess, epoch):
		current_values = {}

		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				v = sess.run(
							self.PGNetwork[task].critic.values, 
							feed_dict={
								self.PGNetwork[task].critic.inputs: [self.env.cv_state_onehot[state_index]],
							})
			
				current_values[x,y,task] = v.ravel().tolist()[0]
						
		
		# if (epoch+1) % self.plot_model == 0 or epoch == 0:
		# 	self.plot_figure.plot(current_values, epoch + 1)

		return current_values

	def roll_out(self, current_policy, current_values, task):
		mb_states, mb_actions, mb_rewards, mb_values, mb_dones, mb_redundants = [[] for _ in range(6)]

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

			next_states, rewards, dones, redundants = self.envs_task[task].step(actions)

			mb_rewards.append(rewards)
			mb_dones.append(dones)
			mb_actions.append(actions)
			mb_states.append(states)
			mb_values.append(values)
			mb_redundants.append(redundants)

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
		mb_redundants = np.array(mb_redundants, dtype = np.float32).T

		self.current_states[task] = next_states

		return mb_states, mb_actions, mb_rewards, mb_values, mb_dones, mb_last_values, mb_redundants

	def _make_batch(self, sess, epoch, current_policy, current_values):

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

			return list(advantages)		

		raw_task_states = [[] for i in range(self.num_task)]
		raw_task_actions = [[] for i in range(self.num_task)]
		task_states = [[] for i in range(self.num_task)]
		task_actions = [[] for i in range(self.num_task)]
		task_returns = [[] for i in range(self.num_task)]
		task_advantages = [[] for i in range(self.num_task)]
		task_redundants = [[] for i in range(self.num_task)]
		rewards_summary = [[] for i in range(self.num_task)]
		# true_advantages = [[] for i in range(self.num_task)]

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
			states, actions, rewards, values, dones, last_values, redundants = self.roll_out(current_policy, current_values, task_idx)
			raw_task_states[task_idx] = np.concatenate(states, 0)
			raw_task_actions[task_idx]= np.concatenate(actions, 0)


			for ep_idx, ep_states in enumerate(states):
				task_states[task_idx] += [self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]]  for s in ep_states]
				task_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[ep_idx]]
				task_redundants[task_idx] += list(redundants[ep_idx])
				rewards_summary[task_idx].append(list(rewards[ep_idx]))
				# true_advantages[task_idx] += [self.env.advs[task_idx][s[0], s[1], a] for (s, a) in zip(ep_states, actions[ep_idx])]
						
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

		share_observations = [[] for _ in range(self.num_task)]
		share_actions = [[] for _ in range(self.num_task)]
		share_advantages = [[] for _ in range(self.num_task)]

		if self.share_exp:
			assert self.num_task > 1
				
			sharing = {}

			if self.share_cut:
				share_choice = int(epoch < 420)
			else:
				share_choice = np.random.choice([1, 0], p = [self.share_decay ** epoch, 1 - self.share_decay ** epoch])
			
			for task_idx in range(self.num_task):
				sharing[task_idx] = []
				for idx, s in enumerate(raw_task_states[task_idx]):
						
					if self.env.MAP[s[1]][s[0]] == 2:

						act = raw_task_actions[task_idx][idx]	
						importance_weight = np.mean([current_policy[s[0], s[1], tidx, 1][act] for tidx in range(self.num_task)])
						
						if share_choice == 1:
							# and share with other tasks
							for other_task in range(self.num_task):
								if other_task == task_idx:
									continue

								share_observations[other_task].append(self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]])
								share_actions[other_task].append(self.env.cv_action_onehot[act])
								if args.no_iw:
									share_advantages[other_task].append(task_advantages[task_idx][idx])
								else:
									share_advantages[other_task].append(task_advantages[task_idx][idx] * current_policy[s[0], s[1], other_task, 1][act] / importance_weight)

						if not args.no_iw:
							sharing[task_idx].append((idx, current_policy[s[0], s[1], task_idx, 1][act] / importance_weight))
			if not args.no_iw:
				for task_idx in range(self.num_task):
					for idx, iw in sharing[task_idx]:
						# we must multiply the advantages of sharing positions with the importance weight and we do it exactly ONE time
						task_advantages[task_idx][idx] = task_advantages[task_idx][idx] * iw

		return task_states, task_actions, task_returns, task_advantages, task_redundants, rewards_summary, share_observations, share_actions, share_advantages
		
	def ppo_iter(self, states, actions, returns, advantages):
		batch_size = len(states)

		states = np.array(states, dtype = np.float32)
		actions = np.array(actions, dtype = np.float32)
		returns = np.array(returns, dtype = np.float32)
		advantages = np.array(advantages, dtype = np.float32)

		inds = np.arange(batch_size)
		np.random.shuffle(inds)

		for start in range(0, batch_size, self.mini_batch_size):
			rand_ids = inds[start: start + self.mini_batch_size]
			yield list(states[rand_ids]), list(actions[rand_ids]), list(returns[rand_ids]), list(advantages[rand_ids])
        
        

	def ppo_update(self, sess, old_policy, mb_states, mb_actions, mb_returns, mb_advantages, mbshare_states, mbshare_actions, mbshare_advantages, task_idx):
		share_size = len(mbshare_states)
		mbshare_states = np.array(mbshare_states, dtype = np.float32)
		mbshare_actions = np.array(mbshare_actions, dtype = np.float32)
		mbshare_advantages = np.array(mbshare_advantages, dtype = np.float32)

		for _ in range(self.ppo_epochs):
			for states, actions, returns, advantages in self.ppo_iter(mb_states, mb_actions, mb_returns, mb_advantages):
				permute = np.random.choice(range(share_size), min(self.mini_batch_size, share_size), replace = False)

				actor_states = states + list(mbshare_states[permute])
				actor_actions = actions + list(mbshare_actions[permute])
				old_probs = []
				for onehot_state, onehot_action in zip(actor_states, actor_actions):
					state_index = np.where(onehot_state == 1)[0][0]
					action = np.where(onehot_action == 1)[0][0]

					(x, y) = self.env.state_space[state_index]
					assert self.env.state_to_index[y][x] == state_index

					old_probs.append(old_policy[x, y, task_idx, 1][action])

				old_neg_log_probs = - np.log(old_probs)
				
				self.PGNetwork[task_idx].learn_ppo(sess, 
												actor_states = actor_states, 
												critic_states = states,
												actions = actor_actions, 
												returns = returns, 
												advantages = advantages + list(mbshare_advantages[permute]), 
												old_neg_log_probs = old_neg_log_probs)



	def train(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)

		sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		total_samples = {}
		for epoch in range(self.num_epochs):
			print('epoch {}/{}'.format(epoch+1, self.num_epochs), end = '\r', flush = True)
			
			current_policy = self._prepare_current_policy(sess, epoch)
			current_values = self._prepare_current_values(sess, epoch)

			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states, mb_actions, mb_returns, mb_advantages, mb_redundants, rewards, \
			mbshare_states, mbshare_actions, mbshare_advantages = self._make_batch(sess, epoch, current_policy, current_values)
			#---------------------------------------------------------------------------------------------------------------------#	

			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			sum_dict = {}
			for task_idx in range(self.num_task):
				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx])

				self.ppo_update(sess, 
							current_policy,  
							mb_states[task_idx], 
							mb_actions[task_idx], 
							mb_returns[task_idx], 
							mb_advantages[task_idx],
							mbshare_states[task_idx],
							mbshare_actions[task_idx],
							mbshare_advantages[task_idx], 
							task_idx)
				
				# correct_adv = 0
				# for (estimated_adv, true_adv) in zip(mb_advantages[task_idx], true_advantages[task_idx]):
				# 	if (estimated_adv > 0 and true_adv > 0) or (estimated_adv < 0 and true_adv < 0):
				# 		correct_adv += 1

				sum_dict[self.PGNetwork[task_idx].mean_redundant] = np.mean([re for re in mb_redundants[task_idx] if not np.isnan(re)])
				sum_dict[self.PGNetwork[task_idx].mean_reward] = np.sum(np.concatenate(rewards[task_idx])) / len(rewards[task_idx])

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
			# if epoch % self.save_model == 0:
			# 	saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		