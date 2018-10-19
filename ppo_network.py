import numpy as np
import tensorflow as tf

from utils import openai_entropy, mse, LearningRateDecay

class Actor():
    def __init__(self, state_size, action_size, reuse = False):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope('Actor' if not reuse else "ShareLatent"):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.actions = tf.placeholder(tf.int32, [None, self.action_size])
            self.advantages = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 256], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([256], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Actions"):
            self.W_fc2 = self._fc_weight_variable([256, self.action_size], name = "W_fc2")
            self.b_fc2 = self._fc_bias_variable([self.action_size], 256, name = "b_fc2")

        self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2

        self.pi = tf.nn.softmax(self.logits)
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.actions)

        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)


class Critic():
    def __init__(self, state_size, reuse = False):
        self.state_size = state_size

        with tf.variable_scope('Critic' if not reuse else "ShareLatent" , reuse  = reuse):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.returns = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 256], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([256], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Value", reuse = False):
            self.W_fc2 = self._fc_weight_variable([256, 1], name = "W_fc3")
            self.b_fc2 = self._fc_bias_variable([1], 256, name = "b_fc3")

            self.values = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
   
        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            
    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)


class A2C_PPO():
    def __init__(self, 
                name, 
                state_size, 
                action_size, 
                entropy_coeff, 
                value_function_coeff, 
                max_gradient_norm, 
                alpha, 
                epsilon,
                clip_param = 0.2, 
                learning_rate = None, 
                decay = False, 
                reuse = False):

        self.max_gradient_norm  = max_gradient_norm
        self.entropy_coeff = entropy_coeff
        self.value_function_coeff = value_function_coeff
        self.state_size = state_size
        self.action_size = action_size
        self.reuse = reuse
        self.alpha = alpha
        self.epsilon = epsilon

        # Add this placeholder for having this variable in tensorboard
        self.mean_reward = tf.placeholder(tf.float32)
        self.mean_redundant = tf.placeholder(tf.float32)
        self.aloss_summary = tf.placeholder(tf.float32)
        self.ploss_summary = tf.placeholder(tf.float32)
        self.vloss_summary = tf.placeholder(tf.float32)
        self.entropy_summary = tf.placeholder(tf.float32)
        self.steps_per_ep = tf.placeholder(tf.float32)
        
        with tf.variable_scope(name):
            self.actor = Actor(state_size = self.state_size, action_size = self.action_size, reuse = self.reuse)
            self.critic = Critic(state_size = self.state_size, reuse = self.reuse)

        self.learning_rate = tf.placeholder(tf.float32, [])
        self.fixed_lr = learning_rate
        self.decay = decay

        self.old_neg_log_prob = tf.placeholder(tf.float32, [None, ])

        self.ratio = tf.exp(self.old_neg_log_prob - self.actor.neg_log_prob)

        surr1 = - self.ratio * self.actor.advantages
        surr2 = - tf.clip_by_value(self.ratio, 1.0 - clip_param, 1.0 + clip_param) * self.actor.advantages

        with tf.variable_scope(name + "/loss/ploss"):
            self.policy_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

        with tf.variable_scope(name + "/loss/vloss"):
            self.value_loss = tf.reduce_mean(mse(tf.squeeze(self.critic.values), self.critic.returns))

        with tf.variable_scope(name + "/loss/entropy"):
            self.entropy = tf.reduce_mean(openai_entropy(self.actor.logits))

        with tf.variable_scope(name + "/loss"):
            self.total_loss = self.policy_loss + self.value_loss * self.value_function_coeff - self.entropy * self.entropy_coeff
        
        self.params = self.find_trainable_variables(name)
        self.grads = tf.gradients(self.total_loss, self.params)
        if self.max_gradient_norm is not None:
            self.grads, grad_norm = tf.clip_by_global_norm(self.grads, self.max_gradient_norm)

        # Apply Gradients
        self.grads = list(zip(self.grads, self.params))
        optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, decay=self.alpha, epsilon=self.epsilon)

        self.train_opt = optimizer.apply_gradients(self.grads)

    def set_lr_decay(self, lr_rate, nvalues):
        self.learning_rate_decayed = LearningRateDecay(v = lr_rate,
                                                       nvalues = nvalues,
                                                       lr_decay_method='linear')
        print("Learning rate decay-er has been set up!")

    def find_trainable_variables(self, key, printing = False):
        with tf.variable_scope(key):
            variables = tf.trainable_variables(key)
            if printing:
                print(len(variables), variables)
            return variables

    def learn_ppo(self, sess, actor_states, critic_states, actions, returns, advantages, old_neg_log_probs):
        if self.decay:
            for i in range(len(actor_states)):
                current_learning_rate = self.learning_rate_decayed.value()
        else:
            current_learning_rate = self.fixed_lr

        feed_dict = {
                        self.actor.inputs: actor_states, # --> actor.logits
                        self.actor.actions: actions, # + logits --> actor.neg_log_prob 
                        self.actor.advantages: advantages,
                        
                        self.critic.inputs: critic_states, # --> critic.values
                        self.critic.returns: returns, # + critic.values --> value loss
                        self.old_neg_log_prob: old_neg_log_probs, # + actor.neg_log_prob --> policy loss

                        self.learning_rate: current_learning_rate,
                    }

        try:
            policy_loss, value_loss, policy_entropy, total_loss, _ = sess.run(
                [self.policy_loss, self.value_loss, self.entropy, self.total_loss, self.train_opt],
                feed_dict = feed_dict
            )

        except ValueError:
            import sys
            print("States: ", states)
            print("Returns: ", returns)
            print("Actions: ", actions)
            print("Advantages: ", advantages)
            sys.exit()

        return policy_loss, value_loss, policy_entropy, total_loss
