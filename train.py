import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import os 
import time
import argparse
import sys

from datetime import datetime
from network import *
from multitask_policy import MultitaskPolicy
from runner import Runner

from env.terrain import Terrain

ask = input('Would you like to create new log folder?Y/n ')
if ask == '' or ask.lower() == 'y':
	log_name = input("New folder's name: ")
	TIMER = str(datetime.now()).replace(' ', '_').replace(":", '-').split('.')[0] + "_" + str(log_name).replace(" ", "_")
else:
	dirs = [d for d in os.listdir('logs/') if os.path.isdir('logs/' + d)]
	TIMER = sorted(dirs, key=lambda x: os.path.getctime('logs/' + x), reverse=True)[0]

def training(args):
	tf.reset_default_graph()

	env = Terrain(args.map_index, args.use_laser)
	policies = []
	oracle_network = {}
	for i in range(args.num_task):
		policy_i = A2C(
						name 					= 'A2C_' + str(i),
						state_size 				= env.cv_state_onehot.shape[1], 
						action_size				= env.action_size,
						entropy_coeff 			= args.ec,
						value_function_coeff 	= args.vc,
						max_gradient_norm		= args.max_gradient_norm,
						alpha 					= args.alpha,
						epsilon					= args.epsilon,
						joint_loss				= args.joint_loss,
						learning_rate			= args.lr,
						decay 					= args.decay,
						reuse					= bool(args.share_latent)
						)

		if args.decay:
			policy_i.set_lr_decay(args.lr, args.num_epochs * args.num_episode * args.num_iters)
		
		print("\nInitialized network {}, with {} trainable weights.".format('A2C_' + str(i), len(policy_i.find_trainable_variables('A2C_' + str(i), True))))
		policies.append(policy_i)

	for i in range(args.num_task - 1):
		for j in range(i+1, args.num_task):
			oracle_network[i, j] = ZNetwork(
										state_size = env.cv_state_onehot.shape[1],
										action_size = 2,
										learning_rate = args.lr,
										name = 'oracle_{}_{}'.format(i, j)
									)

	variables = tf.trainable_variables()
	print("Initialized networks, with {} trainable weights.".format(len(variables)))
		
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)

	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	log_folder = 'logs/' + TIMER

	suffix = []
	for arg in vars(args):
		exclude = ['num_test', 'map_index', 'plot_model', 'save_model', 'num_epochs', 'max_gradient_norm', 'alpha', 'epsilon', 'joint_loss']
		if arg in exclude:
			continue

		boolean = ['share_exp', 'share_latent', 'use_laser', 'use_gae', 'immortal', 'decay', 'noise_argmax', 'oracle']
		if arg in boolean:
			if getattr(args, arg) != 1:
				continue
			else:
				suffix.append(arg)
				continue

		if arg in ['no_iw'] and getattr(args, 'share_exp') == 0:
			continue

		if arg in ['ec', 'vc'] and getattr(args, 'joint_loss') == 0:
			continue

		suffix.append(arg + "_" + str(getattr(args, arg)))

	suffix = '-'.join(suffix)

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)

	if os.path.isdir(os.path.join(log_folder, suffix)):
		print("Log folder already exists. Continue training ...")
		test_time = len(os.listdir(os.path.join(log_folder, suffix)))
	else:
		os.mkdir(os.path.join(log_folder, suffix))
		test_time = 0
	
	if test_time == 0:
		writer = tf.summary.FileWriter(os.path.join(log_folder, suffix))
	else:
		writer = tf.summary.FileWriter(os.path.join(log_folder, suffix))
	
	test_name =  "map_" + str(args.map_index) + "_test_" + str(test_time)
	tf.summary.scalar(test_name + "/rewards", tf.reduce_mean([policy.mean_reward for policy in policies], 0))
	tf.summary.scalar(test_name + "/vloss", tf.reduce_mean([policy.vloss_summary for policy in policies], 0))
	tf.summary.scalar(test_name + "/redundant_steps", tf.reduce_mean([policy.mean_redundant for policy in policies], 0))

	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
										map_index 			= args.map_index,
										policies 			= policies,
										oracle_network		= oracle_network,
										writer 				= writer,
										write_op 			= write_op,
										num_task 			= args.num_task,
										num_iters 			= args.num_iters,
										num_episode 		= args.num_episode,
										num_epochs			= args.num_epochs,
										gamma 				= 0.99,
										lamb				= 0.96,
										plot_model 			= args.plot_model,
										save_model 			= args.save_model,
										save_name 			= test_name + "_" + suffix,
										share_exp 			= args.share_exp,
										oracle				= args.oracle,
										use_laser			= args.use_laser,
										use_gae				= args.use_gae,
										noise_argmax		= args.noise_argmax,
										timer 				= TIMER
									)

	multitask_agent.train(sess, os.path.join(log_folder, suffix, 'checkpoints'))
	sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--num_test', nargs='?', type=int, default = 1, 
						help='Number of test to run')
	parser.add_argument('--map_index', nargs='?', type=int, default = 2, 
						help='Index of map'),
	parser.add_argument('--num_task', nargs='?', type=int, default = 1, 
    					help='Number of tasks to train on')
	parser.add_argument('--immortal', nargs='?', type=int, default = 0, 
    					help='Whether the agent dies when hitting the wall')
	parser.add_argument('--share_exp', nargs='?', type=int, default = 0, 
    					help='Whether to turn on sharing samples on training')
	parser.add_argument('--share_latent', nargs='?', type=int, default = 0,
						help='Whether to join the latent spaces of actor and critic')
	parser.add_argument('--num_episode', nargs='?', type=int, default = 10,
    					help='Number of episodes to sample in each epoch')
	parser.add_argument('--num_iters', nargs='?', type=int, default = None,
						help='Number of steps to be sampled in each episode')
	parser.add_argument('--lr', nargs='?', type=float, default = 0.005,
						help='Learning rate')
	parser.add_argument('--use_laser', nargs='?', type=int, default = 0,
						help='Whether to use laser as input observation instead of one-hot vector')
	parser.add_argument('--use_gae', nargs='?', type=int, default = 1,
						help='Whether to use generalized advantage estimate')
	parser.add_argument('--num_epochs', nargs='?', type=int, default = 2000,
						help='Number of epochs to train')
	parser.add_argument('--oracle', nargs='?', type=int, default = 0,
						help='Whether to use oracle map when sharing')
	parser.add_argument('--ec', nargs='?', type=float, default = 0.01,
						help='Entropy coeff in total loss')
	parser.add_argument('--vc', nargs='?', type=float, default = 0.5,
						help='Value loss coeff in total loss')
	parser.add_argument('--max_gradient_norm', nargs='?', type=float, default = None,
						help='')
	parser.add_argument('--alpha', nargs='?', type=float, default = 0.99,
						help='Optimizer params')
	parser.add_argument('--epsilon', nargs='?', type=float, default = 1e-5,
						help='Optimizer params')
	parser.add_argument('--plot_model', nargs='?', type=int, default = 1000,
						help='Plot interval')
	parser.add_argument('--decay', nargs='?', type=int, default = 0,
						help='Whether to decay the learning_rate')
	parser.add_argument('--noise_argmax', nargs='?', type=int, default = 0,
						help='Whether touse noise argmax in action sampling')
	parser.add_argument('--joint_loss', nargs='?', type=int, default = 0,
						help='Whether to join loss function')
	parser.add_argument('--save_model', nargs='?', type=int, default = 500,
						help='Saving interval')

	args = parser.parse_args()

	start = time.time()
	for i in range(args.num_test):
		training(args)

	print("Done in {} minutes".format((time.time() - start)/60))

