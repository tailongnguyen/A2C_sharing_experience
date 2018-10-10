# from common.multiprocessing_env import SubprocVecEnv
# from common.terrain import Terrain

# import random
# import numpy as np 

# num_envs = 3
# nsteps = 4

# def make_env(**kwargs):
#     def _thunk():
#         return Terrain(**kwargs)

#     return _thunk
        
# def actor(states, action_space):
#     return [np.random.choice(range(action_space)) for _ in range(len(states))]

# def critic(states):
#     return [random.random() for _ in range(states.shape[0])]

# envs = [make_env(map_index = 4, use_laser = True, immortal = False, task = 0) for _ in range(num_envs)]
# envs += [make_env(map_index = 4, use_laser = True, immortal = False, task = 1) for _ in range(num_envs)]

# envs = SubprocVecEnv(envs)


# init_states = envs.reset()
# print(init_states)

# states = [init_states]
# mb_actions = []
# for i in range(nsteps):
#     actions = actor(init_states, envs.action_space)
#     next_states, rewards, dones = envs.step(actions)
#     states.append(next_states)
#     mb_actions.append(actions)

# print(states)
# mb_states = np.concatenate(states, 1).reshape((len(envs), -1, 2))
# print(mb_states)
# # print(np.array(states).reshape(-1, len(envs), 2))
# # print(np.transpose(np.array(states).reshape(-1, len(envs), 2), axes = (1, 2)))
# # print(mb_actions)
# # print(np.array(mb_actions).T)

import numpy as np
def _generalized_advantage_estimate(rewards, dones, values, last_value, gamma = 0.99, lamb = 0.96):
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

def _GAE(episode_rewards, values, last_value, gamma = 0.99, lamda=0.96):
	ep_GAE = np.zeros_like(episode_rewards)
	TD_error = np.zeros_like(episode_rewards)

	next_value = 0.0
	if episode_rewards[-1] == 1:
		next_value = 0.0
	else:
		next_value = last_value

	for i in reversed(range(len(episode_rewards))):
		TD_error[i] = episode_rewards[i]+gamma*next_value - values[i]
		next_value = values[i]

	ep_GAE[len(episode_rewards)-1] = TD_error[len(episode_rewards)-1]
	weight = gamma*lamda
	for i in reversed(range(len(episode_rewards)-1)):
		ep_GAE[i] += TD_error[i]+weight*ep_GAE[i+1]

	return ep_GAE.tolist()	

rewards = [0.4, -0.1, 0.5, 0.2, 1]
values = [0.1, -0.2, 0.3, 0.1, -0.1]
dones = [0, 0, 0, 0, 1]
last_value = 0.15

print(_generalized_advantage_estimate(rewards, dones, values, last_value))
print(_GAE(rewards, values, last_value))