from network import *
from env.terrain import Terrain

model_dir = "/home/yoshi/HMI/current-project/logs/num_task_1-num_episode_12-num_iters_50-lr_0.005-use_gae/A2C_0.cpt"
env = Terrain(2, False)

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

policy = A2C(
		name 					= 'A2C_' + str(0),
		state_size 				= env.cv_state_onehot.shape[1], 
		action_size				= env.action_size,
		entropy_coeff 			= 0.01,
		value_function_coeff 	= 0.5,
		max_gradient_norm		= None,
		alpha 					= 0.1,
		epsilon					= 0.9,
		joint_loss				= False,
		learning_rate			= 0.005,
		decay 					= False,
		reuse					= False
		)

policy.save_model(sess, model_dir, saver)
