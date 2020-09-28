import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.signal
import threading
import time
import os
import multiprocessing
import itertools
import sys
from python_modules import WCST_env as We #need env file ('WCST_env.py'); here it is in directory python_modules/

#####Helper fxns
#Function to specify which gpu to use
def set_gpu(gpu, frac):
    """
    Function to specify which GPU to use

    gpu: string for gpu (i.e. '0')
    frac: memory fraction (i.e. 0.3 for 30%)

    returns tf sess config

    example usage:
    	sess = tf.Session(config=tf.ConfigProto(gpu_options=set_gpu('0', 0.5)))
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options

#create fxn that allows worker to make a working copy of the central_network
def get_network_vars(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope) #get the values of the collection from from_scope
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope) #get values from to_scope

    op_holder = [] #list to hold the from_scope values
    for from_var,to_var in zip(from_vars,to_vars): #for each corresponding pair of values in from_scope and to_scope
        op_holder.append(to_var.assign(from_var)) #assign the from_scope value to the to_scope value and append it to op_holder 
    return op_holder #returns the from_scope values in a list

#for LESIONS - set some weights to zero; takes the central network weights and sets ones that should be zero to zero
def resetzero_network_vars(from_scope,to_scope):
    global v_mask, pi_mask, input_w_mask
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope) #get the values of the collection from from_scope
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope) #get values from to_scope

    resetfrom_vars_1 = from_vars
    resetfrom_vars_1[15] = tf.multiply(from_vars[15],v_mask) #sets masked weights to zero
    resetfrom_vars_1[14] = tf.multiply(from_vars[14],pi_mask)
    resetfrom_vars_1[12] = tf.multiply(from_vars[12],input_w_mask)

    op_holder = [] #list to hold the from_scope values
    for from_var,to_var in zip(resetfrom_vars_1,to_vars): #for each corresponding pair of values in from_scope and to_scope
        op_holder.append(to_var.assign(from_var)) #assign the from_scope value to the to_scope value and append it to op_holder 
    return op_holder #returns the from_scope values in a list

def worker_choose_action(policy_rec):
	action_chosen_index = np.argmax(policy_rec) #choose action with highest prob
	action_chosen = np.zeros(policy_rec.shape[1])
	action_chosen[action_chosen_index] = 1
	return(action_chosen) #1-hot action vector

def worker_act(env, state, action):
	r_cur, s_new, ep_term, correct_ind, action = We.do_action_in_environment(env,state,action)
	return r_cur, s_new, ep_term, correct_ind, action

def worker_decide(decision,pr_list):
	chosen_policy = pr_list[np.argmax(decision)] #in dnet n1 is 0 and n2 is 1; action is 1-hot so n1 is [1,0], n2 is [0,1]
	action_chosen = worker_choose_action(chosen_policy)
	return(action_chosen) #1-hot action vector

#worker has a fxn to discount rewards - he'll save rewards as is until he needs to calculate gradients at the end of
#the episode, then he'll calculate the discounted rewards looking from each state to end of ep
#and use these as the rewards for the advantage calculation in defining the loss function (policy and value loss components
#both need the advantage)

# Discounting function used to calculate discounted returns in the form below when given [r0, r1, r2, r3, ..., rn-1, Vb]
# [y(0), y(1), y(2), ...], where
# e.g. y(0) = r_0 + gamma*r_1 + gamma^2 *r_2 + gamma^3 *r_3 + ... + gamma^(n-1)*r_n-1 + gamma^n*Vb
# e.g. y(2) = r_2 + gamma*r_3 + gamma^2 *r_4 + gamma^3 *r_5 + ... + gamma^(n-3)*r_n-1 + gamma^(n-2)*Vb
# last term is y(n) = Vb
def worker_discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

###

#central network's goal is to optimize to task
#recruit bunch of workers to help
	#work process: each worker takes a copy of the network as they get to work; go explore in environment using the copy
	#based on their experience in environment, calculate updates they would make
	#send the updates back to the central network - central network applies them; applies from each worker
	#when worker send in update recommendations, worker also tosses copy of network and takes new copy of newest version
	#repeat until told to stop

class the_network():
	def __init__(self,state_space, action_space, name, trainer):
		global NETSZ_D, NETSZ_E, LTYPE, rnnout_Lm, v_mask
		with tf.variable_scope(name):
			self.name = name #defines if its the central_network or a worker's working_copy_network

			#placeholder for inputs
			self.inputs = tf.placeholder(shape=[None,state_space],dtype=tf.float32) #environmental inputs (first just inputs from state status)
			
			#if no lesion, LTYPE = 0
			if LTYPE==1: #input of previous reward is lost
				input_L = tf.constant([1,1,1,1,0],shape=[1,state_space],dtype=tf.float32)
			elif LTYPE==2: #input of previous reward is lost
				input_L = tf.constant([1,1,1,0,1],shape=[1,state_space],dtype=tf.float32)
			elif LTYPE==3: #input of previous reward is lost
				input_L = tf.constant([1,1,1,0,0],shape=[1,state_space],dtype=tf.float32)
			else:
				input_L = tf.constant([1,1,1,1,1],shape=[1,state_space],dtype=tf.float32) #aka don't do anything to inputs
			self.inputs_p = tf.multiply(input_L, self.inputs)
			rnn_in = tf.expand_dims(self.inputs_p,[0])

			# network 1 (n1)
			with tf.variable_scope('n1'):
				#make the LSTM - receives from input and outputs to two fully connected layers, 1 for policy and 1 for value
				n1_sizeoflstmcell = NETSZ_E
				n1_lstm = tf.contrib.rnn.BasicLSTMCell(n1_sizeoflstmcell,state_is_tuple=True) #inputs feed to lstm cell
				#define the lstm states ct and ht
				n1_c_start = np.zeros((1,n1_lstm.state_size.c), np.float32)
				n1_h_start = np.zeros((1,n1_lstm.state_size.h), np.float32)
				self.n1_lstm_state_init = [n1_c_start, n1_h_start] #this is an attribute of self because it will be called when a network is made
				n1_c_in = tf.placeholder(tf.float32, [1,n1_lstm.state_size.c])
				n1_h_in = tf.placeholder(tf.float32, [1,n1_lstm.state_size.h])
				self.n1_state_in = (n1_c_in, n1_h_in) # attribute of self because it will be called when using the network to predict
				n1_state_in = tf.nn.rnn_cell.LSTMStateTuple(n1_c_in, n1_h_in) #form of c and h that can be passed back into the LSTM
				n1_batch_size = tf.shape(self.inputs)[:1]
				#connect inputs to lstm and parse lstm outputs
				n1_lstm_outputs, n1_lstm_state = tf.nn.dynamic_rnn(n1_lstm,rnn_in,initial_state=n1_state_in, sequence_length=n1_batch_size)
				n1_lstm_c, n1_lstm_h = n1_lstm_state
				self.n1_state_out = (n1_lstm_c[:1, :], n1_lstm_h[:1, :]) #will call this to keep track of c and h states
				n1_rnn_out = tf.reshape(n1_lstm_outputs, [-1,n1_sizeoflstmcell]) #output of each of the units in the LSTM

				#fully connected layers at end to give policy and value
				self.n1_policy_layer_output = slim.fully_connected(n1_rnn_out,action_space,
					activation_fn=tf.nn.softmax,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)
				
				self.n1_value_layer_output = slim.fully_connected(n1_rnn_out,1,
					activation_fn=None,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)

				if name != 'central_network':
					#to calc gradients need
						#state, action, policy, value, discounted_reward
					#to get gradients:
						#will give network s, ch_state_in -> these will generate
						#self.policy_layer_output and self.value_layer_output
						#then also give action, discounted_R

					self.n1_A = tf.placeholder(shape=[None,action_space],dtype=tf.float32) #1-hot action taken from this state
					self.n1_R = tf.placeholder(shape=[None,1],dtype=tf.float32) #reward estimate of this state based on rest of episode experience: rt + gamma**1 * rt+1 +...+gamma**k * V(s_end)

					n1_selection_from_policy = tf.reduce_sum(self.n1_policy_layer_output * self.n1_A, [1]) #this is pi(A,S)
					n1_sfp = tf.reshape(n1_selection_from_policy,[-1,1]) #makes it (batch_size, 1)

					n1_advantage = self.n1_R - self.n1_value_layer_output
					#define loss function: Total_loss = Policy_loss + Value_loss + Entropy_loss
					n1_Policy_loss = - tf.log(n1_sfp + 1e-10) * tf.stop_gradient(n1_advantage)
						#advantage tells the magnitude that you should move toward this policy choice
						#movement of weights toward the policy taken, i.e. this policy maximizes the reward from this action so move toward it
						#aka maximize this policy in this step
						#aka - log (policy), i.e. minimize the negative
					n1_Value_loss = tf.square(n1_advantage)
					#entropy term to encourage exploration: H = - sum(p * log p); this term will  be subtracted from total loss function
					# so that if entropy is large (big H), the total loss will be lower
					n1_Entropy_loss = - tf.reduce_sum(self.n1_policy_layer_output * tf.log(self.n1_policy_layer_output + 1e-10))

					n1_c_V = 0.05
					n1_c_E = 0.05
					n1_Total_loss = n1_Policy_loss + n1_c_V*n1_Value_loss - n1_c_E*n1_Entropy_loss
					self.n1_tl = n1_Total_loss

					#calculate the gradient of the loss function - use this to update th network
					n1_local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/n1')
					self.n1_gradient_loss = tf.gradients(n1_Total_loss,n1_local_vars) #worker will send these gradients (recommended updates) to central_network's gradient_list

					n1_grads_to_apply = self.n1_gradient_loss

					#worker can then apply the gradients to the central_network
					n1_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network/n1')
					self.n1_apply_gradients = trainer.apply_gradients(zip(n1_grads_to_apply,n1_global_vars))

			# network 2 (n2)
			with tf.variable_scope('n2'):
				n2_sizeoflstmcell = NETSZ_E
				n2_lstm = tf.contrib.rnn.BasicLSTMCell(n2_sizeoflstmcell,state_is_tuple=True)
				n2_c_start = np.zeros((1,n2_lstm.state_size.c), np.float32)
				n2_h_start = np.zeros((1,n2_lstm.state_size.h), np.float32)
				self.n2_lstm_state_init = [n2_c_start, n2_h_start]
				n2_c_in = tf.placeholder(tf.float32, [1,n2_lstm.state_size.c])
				n2_h_in = tf.placeholder(tf.float32, [1,n2_lstm.state_size.h])
				self.n2_state_in = (n2_c_in, n2_h_in)
				n2_state_in = tf.nn.rnn_cell.LSTMStateTuple(n2_c_in, n2_h_in)
				n2_batch_size = tf.shape(self.inputs)[:1]
				n2_lstm_outputs, n2_lstm_state = tf.nn.dynamic_rnn(n2_lstm,rnn_in,initial_state=n2_state_in, sequence_length=n2_batch_size)
				n2_lstm_c, n2_lstm_h = n2_lstm_state
				self.n2_state_out = (n2_lstm_c[:1, :], n2_lstm_h[:1, :]) 
				n2_rnn_out = tf.reshape(n2_lstm_outputs, [-1,n2_sizeoflstmcell])

				self.n2_policy_layer_output = slim.fully_connected(n2_rnn_out,action_space,
					activation_fn=tf.nn.softmax,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)
				
				self.n2_value_layer_output = slim.fully_connected(n2_rnn_out,1,
					activation_fn=None,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)

				if name != 'central_network':
					self.n2_A = tf.placeholder(shape=[None,action_space],dtype=tf.float32)
					self.n2_R = tf.placeholder(shape=[None,1],dtype=tf.float32)
					n2_selection_from_policy = tf.reduce_sum(self.n2_policy_layer_output * self.n2_A, [1])
					n2_sfp = tf.reshape(n2_selection_from_policy,[-1,1])
					n2_advantage = self.n2_R - self.n2_value_layer_output
					n2_Policy_loss = - tf.log(n2_sfp + 1e-10) * tf.stop_gradient(n2_advantage)
					n2_Value_loss = tf.square(n2_advantage)
					n2_Entropy_loss = - tf.reduce_sum(self.n2_policy_layer_output * tf.log(self.n2_policy_layer_output + 1e-10))
					
					n2_c_V = 0.05
					n2_c_E = 0.05
					n2_Total_loss = n2_Policy_loss + n2_c_V*n2_Value_loss - n2_c_E*n2_Entropy_loss
					self.n2_tl = n2_Total_loss

					n2_local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/n2')
					self.n2_gradient_loss = tf.gradients(n2_Total_loss,n2_local_vars)
					n2_grads_to_apply = self.n2_gradient_loss
					n2_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network/n2')
					self.n2_apply_gradients = trainer.apply_gradients(zip(n2_grads_to_apply,n2_global_vars))

			# network 3 (n3)
			with tf.variable_scope('n3'):
				n3_sizeoflstmcell = NETSZ_E
				n3_lstm = tf.contrib.rnn.BasicLSTMCell(n3_sizeoflstmcell,state_is_tuple=True)
				n3_c_start = np.zeros((1,n3_lstm.state_size.c), np.float32)
				n3_h_start = np.zeros((1,n3_lstm.state_size.h), np.float32)
				self.n3_lstm_state_init = [n3_c_start, n3_h_start]
				n3_c_in = tf.placeholder(tf.float32, [1,n3_lstm.state_size.c])
				n3_h_in = tf.placeholder(tf.float32, [1,n3_lstm.state_size.h])
				self.n3_state_in = (n3_c_in, n3_h_in)
				n3_state_in = tf.nn.rnn_cell.LSTMStateTuple(n3_c_in, n3_h_in)
				n3_batch_size = tf.shape(self.inputs)[:1]
				n3_lstm_outputs, n3_lstm_state = tf.nn.dynamic_rnn(n3_lstm,rnn_in,initial_state=n3_state_in, sequence_length=n3_batch_size)
				n3_lstm_c, n3_lstm_h = n3_lstm_state
				self.n3_state_out = (n3_lstm_c[:1, :], n3_lstm_h[:1, :])
				n3_rnn_out = tf.reshape(n3_lstm_outputs, [-1,n3_sizeoflstmcell])

				self.n3_policy_layer_output = slim.fully_connected(n3_rnn_out,action_space,
					activation_fn=tf.nn.softmax,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)
				
				self.n3_value_layer_output = slim.fully_connected(n3_rnn_out,1,
					activation_fn=None,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)

				if name != 'central_network':
					self.n3_A = tf.placeholder(shape=[None,action_space],dtype=tf.float32)
					self.n3_R = tf.placeholder(shape=[None,1],dtype=tf.float32)
					n3_selection_from_policy = tf.reduce_sum(self.n3_policy_layer_output * self.n3_A, [1])
					n3_sfp = tf.reshape(n3_selection_from_policy,[-1,1])
					n3_advantage = self.n3_R - self.n3_value_layer_output
					n3_Policy_loss = - tf.log(n3_sfp + 1e-10) * tf.stop_gradient(n3_advantage)
					n3_Value_loss = tf.square(n3_advantage)
					n3_Entropy_loss = - tf.reduce_sum(self.n3_policy_layer_output * tf.log(self.n3_policy_layer_output + 1e-10))

					n3_c_V = 0.05
					n3_c_E = 0.05
					n3_Total_loss = n3_Policy_loss + n3_c_V*n3_Value_loss - n3_c_E*n3_Entropy_loss
					self.n3_tl = n3_Total_loss

					n3_local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/n3')
					self.n3_gradient_loss = tf.gradients(n3_Total_loss,n3_local_vars)
					n3_grads_to_apply = self.n3_gradient_loss
					n3_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network/n3')
					self.n3_apply_gradients = trainer.apply_gradients(zip(n3_grads_to_apply,n3_global_vars))

			with tf.variable_scope('dnet'):
				dnet_sizeoflstmcell = NETSZ_D
				dnet_lstm = tf.contrib.rnn.BasicLSTMCell(dnet_sizeoflstmcell,state_is_tuple=True)
				dnet_c_start = np.zeros((1,dnet_lstm.state_size.c), np.float32)
				dnet_h_start = np.zeros((1,dnet_lstm.state_size.h), np.float32)
				self.dnet_lstm_state_init = [dnet_c_start, dnet_h_start]
				dnet_c_in = tf.placeholder(tf.float32, [1,dnet_lstm.state_size.c])
				dnet_h_in = tf.placeholder(tf.float32, [1,dnet_lstm.state_size.h])
				self.dnet_state_in = (dnet_c_in, dnet_h_in)
				dnet_state_in = tf.nn.rnn_cell.LSTMStateTuple(dnet_c_in, dnet_h_in)
				dnet_batch_size = tf.shape(self.inputs)[:1]
				dnet_lstm_outputs, dnet_lstm_state = tf.nn.dynamic_rnn(dnet_lstm,rnn_in,initial_state=dnet_state_in, sequence_length=dnet_batch_size)
				dnet_lstm_c, dnet_lstm_h = dnet_lstm_state
				self.dnet_state_out = (dnet_lstm_c[:1, :], dnet_lstm_h[:1, :])
				dnet_rnn_out = tf.reshape(dnet_lstm_outputs, [-1,dnet_sizeoflstmcell])

				rnnout_L = tf.constant(rnnout_Lm,shape=[1,dnet_sizeoflstmcell],dtype=tf.float32)
				dnet_rnn_out_p = tf.multiply(rnnout_L, dnet_rnn_out)

				self.dnet_policy_layer_output = slim.fully_connected(dnet_rnn_out_p,3,
					activation_fn=tf.nn.softmax,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)
				
				self.dnet_value_layer_output = slim.fully_connected(dnet_rnn_out_p,1,
					activation_fn=None,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=None)

				if name != 'central_network':
					self.dnet_D = tf.placeholder(shape=[None,3],dtype=tf.float32)
					self.dnet_R = tf.placeholder(shape=[None,1],dtype=tf.float32)
					dnet_selection_from_policy = tf.reduce_sum(self.dnet_policy_layer_output * self.dnet_D, [1])
					dnet_sfp = tf.reshape(dnet_selection_from_policy,[-1,1])
					dnet_advantage = self.dnet_R - self.dnet_value_layer_output
					dnet_Policy_loss = - tf.log(dnet_sfp + 1e-10) * tf.stop_gradient(dnet_advantage)
					dnet_Value_loss = tf.square(dnet_advantage)
					dnet_Entropy_loss = - tf.reduce_sum(self.dnet_policy_layer_output * tf.log(self.dnet_policy_layer_output + 1e-10))
					
					dnet_c_V = 0.05
					dnet_c_E = 0.05
					dnet_Total_loss = dnet_Policy_loss + dnet_c_V*dnet_Value_loss - dnet_c_E*dnet_Entropy_loss
					self.dnet_tl = dnet_Total_loss

					dnet_local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/dnet')
					self.lv = dnet_local_vars
					self.dnet_gradient_loss = tf.gradients(dnet_Total_loss,dnet_local_vars)
					dnet_grads_to_apply = self.dnet_gradient_loss
					dnet_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network/dnet')
					self.dnet_apply_gradients = trainer.apply_gradients(zip(dnet_grads_to_apply,dnet_global_vars))
					self.gv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network')

			#central network isn't held by a worker - doesn't need to calculate loss and gradient; each worker has to do this
			#workers then send the updates they would make to the central network
			#central network uses the gradients to update itself
			#after each episode, worker throws away the old copy of network and takes a copy of the uptodate global network to go work on 

			#this is a section of the network that only workers have - it takes the data they've collected from an episode and
			# uses it to calculate losses and from these, gradients for each parameter (weight)
			# the worker will then send these gradients to the central network and apply them to tune it

#worker can only be created if there is a global the_network object defined called central_network and a global variables
#defined: GLOBAL_EPISODE_COUNTER, STATE_SPACE, ACTION_SPACE, GAMMA, MAX_EPISODE_LEN
class worker():

	def __init__(self,name,trainer,actorenv):
		self.name = 'worker' + str(name)
		self.actorenv = actorenv
		self.env = We.make_new_env(actorenv=self.actorenv) #each worker creates own instance of environment to interact with
		self.trainer = trainer
		self.working_copy_network = the_network(state_space=STATE_SPACE, action_space=ACTION_SPACE, name=self.name, trainer=trainer)
		self.working_copy_network_params = get_network_vars('central_network',self.name)
		self.default_graph = tf.get_default_graph()
		self.assesser = []

	#when episode is done, worker gathers data and processes it
	#passes processed data to his own network to calculate gradients
	#applies those gradients to the central_network
	#bootstrap value is 0 if episode ended in terminal state; V(s_n+1) if episode was cut off in state s_n+1
		#i.e. worker was in s_n and did action a_n to move to s_n+1, then episode was cut because exceed max length
	def train(self,training_data,bootstrap_value,gamma,sess):
		global GLOBAL_EPISODE_COUNTER, TO_TRAIN
		#first replace the rewards with the discounted-rewards because this is what the network needs to calc losses
		array_training_data = np.array(training_data)
		step_rewards = [ritem for sublist in array_training_data[:,3] for ritem in sublist] #list of the step by step rewards
		step_rewards = step_rewards + [bootstrap_value]

		discR = worker_discount(step_rewards,gamma)[:-1] #cut of the last value because it was just used to give discounted reward estimate

		discR_listed = [[item] for item in discR]
		array_training_data[:,3] = discR_listed

		stacked_states = np.vstack(array_training_data[:,0])
		stacked_action = np.vstack(array_training_data[:,2])
		stacked_reward = np.vstack(array_training_data[:,3])
		stacked_decision = np.vstack(array_training_data[:,6])

		if (TO_TRAIN=='Experts_n1') | (TO_TRAIN=='Experts_n2') | (TO_TRAIN=='Experts_n3'):
			if self.actorenv==[0]: #if training network 1 on shape-sorting
				feed_dict = {self.working_copy_network.inputs:stacked_states,
					self.working_copy_network.n1_state_in[0]:self.n1_train_rnn_state[0],
					self.working_copy_network.n1_state_in[1]:self.n1_train_rnn_state[1],
					self.working_copy_network.n1_A:stacked_action,
					self.working_copy_network.n1_R:stacked_reward}
				self.n1_train_rnn_state, gl, _ = sess.run([self.working_copy_network.n1_state_out,
					self.working_copy_network.n1_gradient_loss,
					self.working_copy_network.n1_apply_gradients],
					feed_dict=feed_dict)

			elif self.actorenv==[1]: #if training network 2 on color-sorting
				feed_dict = {self.working_copy_network.inputs:stacked_states,
					self.working_copy_network.n2_state_in[0]:self.n2_train_rnn_state[0],
					self.working_copy_network.n2_state_in[1]:self.n2_train_rnn_state[1],
					self.working_copy_network.n2_A:stacked_action,
					self.working_copy_network.n2_R:stacked_reward}
				self.n2_train_rnn_state, gl, _ = sess.run([self.working_copy_network.n2_state_out,
					self.working_copy_network.n2_gradient_loss,
					self.working_copy_network.n2_apply_gradients],
					feed_dict=feed_dict)

			elif self.actorenv==[2]: #if training network 2 on color-sorting
				feed_dict = {self.working_copy_network.inputs:stacked_states,
					self.working_copy_network.n3_state_in[0]:self.n3_train_rnn_state[0],
					self.working_copy_network.n3_state_in[1]:self.n3_train_rnn_state[1],
					self.working_copy_network.n3_A:stacked_action,
					self.working_copy_network.n3_R:stacked_reward}
				self.n3_train_rnn_state, gl, _ = sess.run([self.working_copy_network.n3_state_out,
					self.working_copy_network.n3_gradient_loss,
					self.working_copy_network.n3_apply_gradients],
					feed_dict=feed_dict)

		elif TO_TRAIN=='DNET':
			feed_dict = {self.working_copy_network.inputs:stacked_states,
				self.working_copy_network.dnet_state_in[0]:self.dnet_train_rnn_state[0],
				self.working_copy_network.dnet_state_in[1]:self.dnet_train_rnn_state[1],
				self.working_copy_network.dnet_D:stacked_decision,
				self.working_copy_network.dnet_R:stacked_reward}
			self.dnet_train_rnn_state, gv, _ = sess.run([self.working_copy_network.dnet_state_out,
				self.working_copy_network.gv,
				self.working_copy_network.dnet_apply_gradients],
				feed_dict=feed_dict)

			sess.run(resetzero_network_vars('central_network','central_network'))

		elif TO_TRAIN=='DNETAllExpert':
			coded_decision = [np.argmax(item) for item in stacked_decision] #0,1,2 for n1, n2, n3

			which_newexpert = [0 if item!=0 else 1 for item in coded_decision] #all the n3 decisions get 1, all other get 0
			n1_decision_index = np.argwhere(np.array(which_newexpert,ndmin=2)==1)[:,1]
			which_newexpert = [0 if item!=1 else 1 for item in coded_decision] #all the n3 decisions get 1, all other get 0
			n2_decision_index = np.argwhere(np.array(which_newexpert,ndmin=2)==1)[:,1]
			which_newexpert = [0 if item!=2 else 1 for item in coded_decision] #all the n3 decisions get 1, all other get 0
			n3_decision_index = np.argwhere(np.array(which_newexpert,ndmin=2)==1)[:,1]

			if TrainNewOnly==True:
				if WHICH_DNET=='1e':
					if len(n1_decision_index)!=0:
						#Get states in which n3 was used
						n1_used_states = stacked_states[n1_decision_index,:]
						n1_used_actions = stacked_action[n1_decision_index,:]
						n1_used_rewards = stacked_reward[n1_decision_index,:]

						n1_feed_dict = {self.working_copy_network.inputs:n1_used_states,
							self.working_copy_network.n1_state_in[0]:self.n1_train_rnn_state[0],
							self.working_copy_network.n1_state_in[1]:self.n1_train_rnn_state[1],
							self.working_copy_network.n1_A:n1_used_actions,
							self.working_copy_network.n1_R:n1_used_rewards}
						self.n1_train_rnn_state, n1_gl, _ = sess.run([self.working_copy_network.n1_state_out,
							self.working_copy_network.n1_gradient_loss,
							self.working_copy_network.n1_apply_gradients],
							feed_dict=n1_feed_dict)
				elif WHICH_DNET=='2e':
					if len(n2_decision_index)!=0:
						#Get states in which n3 was used
						n2_used_states = stacked_states[n2_decision_index,:]
						n2_used_actions = stacked_action[n2_decision_index,:]
						n2_used_rewards = stacked_reward[n2_decision_index,:]

						n2_feed_dict = {self.working_copy_network.inputs:n2_used_states,
							self.working_copy_network.n2_state_in[0]:self.n2_train_rnn_state[0],
							self.working_copy_network.n2_state_in[1]:self.n2_train_rnn_state[1],
							self.working_copy_network.n2_A:n2_used_actions,
							self.working_copy_network.n2_R:n2_used_rewards}
						self.n2_train_rnn_state, n2_gl, _ = sess.run([self.working_copy_network.n2_state_out,
							self.working_copy_network.n2_gradient_loss,
							self.working_copy_network.n2_apply_gradients],
							feed_dict=n2_feed_dict)
				elif WHICH_DNET=='3e':
					if len(n3_decision_index)!=0:
						#Get states in which n3 was used
						n3_used_states = stacked_states[n3_decision_index,:]
						n3_used_actions = stacked_action[n3_decision_index,:]
						n3_used_rewards = stacked_reward[n3_decision_index,:]

						n3_feed_dict = {self.working_copy_network.inputs:n3_used_states,
							self.working_copy_network.n3_state_in[0]:self.n3_train_rnn_state[0],
							self.working_copy_network.n3_state_in[1]:self.n3_train_rnn_state[1],
							self.working_copy_network.n3_A:n3_used_actions,
							self.working_copy_network.n3_R:n3_used_rewards}
						self.n3_train_rnn_state, n3_gl, _ = sess.run([self.working_copy_network.n3_state_out,
							self.working_copy_network.n3_gradient_loss,
							self.working_copy_network.n3_apply_gradients],
							feed_dict=n3_feed_dict)

			else:
				#if n3 was called upon, train it on the episodes it was called for
				if len(n1_decision_index)!=0:
					#Get states in which n3 was used
					n1_used_states = stacked_states[n1_decision_index,:]
					n1_used_actions = stacked_action[n1_decision_index,:]
					n1_used_rewards = stacked_reward[n1_decision_index,:]

					n1_feed_dict = {self.working_copy_network.inputs:n1_used_states,
						self.working_copy_network.n1_state_in[0]:self.n1_train_rnn_state[0],
						self.working_copy_network.n1_state_in[1]:self.n1_train_rnn_state[1],
						self.working_copy_network.n1_A:n1_used_actions,
						self.working_copy_network.n1_R:n1_used_rewards}
					self.n1_train_rnn_state, n1_gl, _ = sess.run([self.working_copy_network.n1_state_out,
						self.working_copy_network.n1_gradient_loss,
						self.working_copy_network.n1_apply_gradients],
						feed_dict=n1_feed_dict)
				if len(n2_decision_index)!=0:
					#Get states in which n3 was used
					n2_used_states = stacked_states[n2_decision_index,:]
					n2_used_actions = stacked_action[n2_decision_index,:]
					n2_used_rewards = stacked_reward[n2_decision_index,:]

					n2_feed_dict = {self.working_copy_network.inputs:n2_used_states,
						self.working_copy_network.n2_state_in[0]:self.n2_train_rnn_state[0],
						self.working_copy_network.n2_state_in[1]:self.n2_train_rnn_state[1],
						self.working_copy_network.n2_A:n2_used_actions,
						self.working_copy_network.n2_R:n2_used_rewards}
					self.n2_train_rnn_state, n2_gl, _ = sess.run([self.working_copy_network.n2_state_out,
						self.working_copy_network.n2_gradient_loss,
						self.working_copy_network.n2_apply_gradients],
						feed_dict=n2_feed_dict)
				if len(n3_decision_index)!=0:
					#Get states in which n3 was used
					n3_used_states = stacked_states[n3_decision_index,:]
					n3_used_actions = stacked_action[n3_decision_index,:]
					n3_used_rewards = stacked_reward[n3_decision_index,:]

					n3_feed_dict = {self.working_copy_network.inputs:n3_used_states,
						self.working_copy_network.n3_state_in[0]:self.n3_train_rnn_state[0],
						self.working_copy_network.n3_state_in[1]:self.n3_train_rnn_state[1],
						self.working_copy_network.n3_A:n3_used_actions,
						self.working_copy_network.n3_R:n3_used_rewards}
					self.n3_train_rnn_state, n3_gl, _ = sess.run([self.working_copy_network.n3_state_out,
						self.working_copy_network.n3_gradient_loss,
						self.working_copy_network.n3_apply_gradients],
						feed_dict=n3_feed_dict)

			#train dnet
			dnet_feed_dict = {self.working_copy_network.inputs:stacked_states,
				self.working_copy_network.dnet_state_in[0]:self.dnet_train_rnn_state[0],
				self.working_copy_network.dnet_state_in[1]:self.dnet_train_rnn_state[1],
				self.working_copy_network.dnet_D:stacked_decision,
				self.working_copy_network.dnet_R:stacked_reward}
			self.dnet_train_rnn_state, gv, dnet_gl, _ = sess.run([self.working_copy_network.dnet_state_out,
				self.working_copy_network.gv,
				self.working_copy_network.dnet_gradient_loss,
				self.working_copy_network.dnet_apply_gradients],
				feed_dict=dnet_feed_dict)
					
	def get_experience(self,sess,coord,NUM_TRAIN_EPS=1000,on_ep=True):
		print ("Starting " + self.name)
		global GLOBAL_EPISODE_COUNTER, NUMBER_OF_WORKERS, TO_TRAIN, WHICH_DNET, TRAIN_EP_COUNT, last400ep, train_to_profst, avecorthresh, aveover
		with sess.as_default(), sess.graph.as_default():   #with this session and session graph set to default
			firstpriorenvtype = True
			stop_training = False
			while not coord.should_stop(): #tf.coordinator is passed to each worker thread that is started; it coordinates threads 
				#get copy of the uptodate central_network parameters
				sess.run(resetzero_network_vars('central_network','central_network'))
				sess.run(self.working_copy_network_params)
				#now worker network is same as uptodate central_network

				training_data = []
				#begin episode
				#if its the first time, get a random envtype to be the t-1 envtype; otherwise give it the last envtype so it picks a different one
				if firstpriorenvtype==True:
					prevenvtype = np.random.choice(np.array([0,1,2]),p=[1/3,1/3,1/3])
					firstpriorenvtype = False
				else:
					prevenvtype = self.env.envtype

				self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',training_deck_indices=training_deck_indices) #for each ep make a new env obj to get new baseline probs of A and B
				start_state = We.get_start_state_from_env(self.env)
				s_cur = start_state
				if (TO_TRAIN=='Experts_n1') | (TO_TRAIN=='Experts_n2') | (TO_TRAIN=='Experts_n3'):
					if self.actorenv==[0]:
						n1_ch_state_in = self.working_copy_network.n1_lstm_state_init #defines ch_state_in as zeros
						self.n1_train_rnn_state = n1_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
						ep_steps = 0
						num_correct = 0
						if on_ep==True:
							ep_term = False
							wholess = 0
							thanwho = 3
						else:
							wholess = 0
							thanwho = 10
						while wholess < thanwho:
							#keep track of [s_cur,a_cur,r_cur,s_new,ep_term]
							#feed st to network to get policy and value output
							# policy_rec, value_pred, ch_state_out = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)
							n1_policy_rec, n1_ch_state_out = sess.run([self.working_copy_network.n1_policy_layer_output,
								self.working_copy_network.n1_state_out],
								feed_dict={self.working_copy_network.inputs:s_cur,
								self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
								self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1]})
									
							#choose action based on policy_rec
							d_cur = np.zeros(3)
							a_cur = worker_choose_action(n1_policy_rec) #a_cur is the 1-hot action vector

							r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
							#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

							new_step_in_environment = [s_cur, n1_ch_state_in, a_cur, r_cur, s_new, ep_term, d_cur]
							training_data.append(new_step_in_environment) #this is the data to calculate gradients with

							s_cur = s_new
							n1_ch_state_in = n1_ch_state_out
							ep_steps += 1
							num_correct += correct_ind
							if (on_ep==True) & (ep_term==True):
								wholess += 1
							elif on_ep==False:
								wholess += 1

							#check if max episode length has been reached
							if wholess == thanwho:
								#use s_cur and ch_state_in to get v(s_cur)
								#_, value_pred, _ = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)
								value_pred = sess.run([self.working_copy_network.n1_value_layer_output],
									feed_dict={self.working_copy_network.inputs:s_cur,
									self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
									self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1]})
								bootstrap_value = value_pred #this is scalar passed to train() below
							
							if ep_term==True:
								if self.name=='workern10':
									if train_to_profst==True: #if training until proficient or stable
										last400ep = np.append(last400ep,ep_steps)
										if len(last400ep)>=10:
											last10ep = last400ep[-10:] #get last 10 episodes
											if len(np.where(last10ep<15)[0])==10: #if last 10 episodes were less than 15 steps, proficiency achieved: stop training
												stop_training = True
										if len(last400ep) > 400: #if >400 points can test for stability criteria
											last400ep = last400ep[-400:] #pop off first one so it is len=400
											if np.abs(np.mean(last400ep[-200:])-np.mean(last400ep[0:200]))<1: #if diff between mean of last 200 and previous 200 is <1 stability achieved: stop training
												stop_training = True

									if train_assesser_on==True:
										self.assesser.append(ep_steps)
										if len(self.assesser)>aveover:
											self.assesser.pop(0)
											if np.mean(self.assesser)<avecorthresh:
												stop_training = True

								ep_steps = 0
								num_correct = 0
								prevenvtype = self.env.envtype
								self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',set_start_state=s_cur,training_deck_indices=training_deck_indices)

	                                        #when episode==done
					        #train processes the training data then runs it through the worker's network to calculate gradients,
					        #calculates gradients, then takes these to the central_network and uses them to update the central_network

					elif self.actorenv==[1]:
						n2_ch_state_in = self.working_copy_network.n2_lstm_state_init #defines ch_state_in as zeros
						self.n2_train_rnn_state = n2_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
						ep_steps = 0
						num_correct = 0
						if on_ep==True:
							ep_term = False
							wholess = 0
							thanwho = 3
						else:
							wholess = 0
							thanwho = 10
						while wholess < thanwho:
							n2_policy_rec, n2_ch_state_out = sess.run([self.working_copy_network.n2_policy_layer_output,
								self.working_copy_network.n2_state_out],
								feed_dict={self.working_copy_network.inputs:s_cur,
								self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
								self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1]})
									
							#choose action based on policy_rec
							d_cur = np.zeros(3)
							a_cur = worker_choose_action(n2_policy_rec) #a_cur is the 1-hot action vector

							r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)

							new_step_in_environment = [s_cur, n2_ch_state_in, a_cur, r_cur, s_new, ep_term, d_cur]
							training_data.append(new_step_in_environment) #this is the data to calculate gradients with

							s_cur = s_new
							n2_ch_state_in = n2_ch_state_out
							ep_steps += 1
							num_correct += correct_ind
							if (on_ep==True) & (ep_term==True):
								wholess += 1 #if on_ep=True then wholess is indicator of when ep is over -> train
							elif on_ep==False:
								wholess += 1 #if on_ep=False then wholess is counting cardpulls

							#check if max episode length has been reached
							if wholess == thanwho:
								value_pred = sess.run([self.working_copy_network.n2_value_layer_output],
									feed_dict={self.working_copy_network.inputs:s_cur,
									self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
									self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1]})
								bootstrap_value = value_pred #this is scalar passed to train() below

							if ep_term==True:
								if (self.name=='workern20'):
									if train_to_profst==True: #if training until proficient or stable
										last400ep = np.append(last400ep,ep_steps)
										if len(last400ep)>=10:
											last10ep = last400ep[-10:] #get last 10 episodes
											if len(np.where(last10ep<15)[0])==10: #if last 10 episodes were less than 15 steps, proficiency achieved: stop training
												stop_training = True
										if len(last400ep) > 400: #if >400 points can test for stability criteria
											last400ep = last400ep[-400:] #pop off first one so it is len=400
											if np.abs(np.mean(last400ep[-200:])-np.mean(last400ep[0:200]))<1: #if diff between mean of last 200 and previous 200 is <1 stability achieved: stop training
												stop_training = True

									if train_assesser_on==True:
										self.assesser.append(ep_steps)
										if len(self.assesser)>aveover:
											self.assesser.pop(0)
											if np.mean(self.assesser)<avecorthresh:
												stop_training = True

								ep_steps = 0
								num_correct = 0
								prevenvtype = self.env.envtype
								self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',set_start_state=s_cur,training_deck_indices=training_deck_indices)

					elif self.actorenv==[2]:
						n3_ch_state_in = self.working_copy_network.n3_lstm_state_init #defines ch_state_in as zeros
						self.n3_train_rnn_state = n3_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
						ep_steps = 0
						num_correct = 0
						if on_ep==True:
							ep_term = False
							wholess = 0
							thanwho = 3
						else:
							wholess = 0
							thanwho = 10
						while wholess < thanwho:
							n3_policy_rec, n3_ch_state_out = sess.run([self.working_copy_network.n3_policy_layer_output,
								self.working_copy_network.n3_state_out],
								feed_dict={self.working_copy_network.inputs:s_cur,
								self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
								self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
									
							#choose action based on policy_rec
							d_cur = np.zeros(3)
							a_cur = worker_choose_action(n3_policy_rec) #a_cur is the 1-hot action vector

							r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)

							new_step_in_environment = [s_cur, n3_ch_state_in, a_cur, r_cur, s_new, ep_term, d_cur]
							training_data.append(new_step_in_environment) #this is the data to calculate gradients with

							s_cur = s_new
							n3_ch_state_in = n3_ch_state_out
							ep_steps += 1
							num_correct += correct_ind
							if (on_ep==True) & (ep_term==True):
								wholess += 1 #if on_ep=True then wholess is indicator of when ep is over -> train
							elif on_ep==False:
								wholess += 1 #if on_ep=False then wholess is counting cardpulls

							#check if max episode length has been reached
							if wholess == thanwho:
								value_pred = sess.run([self.working_copy_network.n3_value_layer_output],
									feed_dict={self.working_copy_network.inputs:s_cur,
									self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
									self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
								bootstrap_value = value_pred #this is scalar passed to train() below

							if ep_term==True:
								if (self.name=='workern30'):
									if train_to_profst==True: #if training until proficient or stable
										last400ep = np.append(last400ep,ep_steps)
										if len(last400ep)>=10:
											last10ep = last400ep[-10:] #get last 10 episodes
											if len(np.where(last10ep<15)[0])==10: #if last 10 episodes were less than 15 steps, proficiency achieved: stop training
												stop_training = True
										if len(last400ep) > 400: #if >400 points can test for stability criteria
											last400ep = last400ep[-400:] #pop off first one so it is len=400
											if np.abs(np.mean(last400ep[-200:])-np.mean(last400ep[0:200]))<1: #if diff between mean of last 200 and previous 200 is <1 stability achieved: stop training
												stop_training = True

									if train_assesser_on==True:
										self.assesser.append(ep_steps)
										if len(self.assesser)>aveover:
											self.assesser.pop(0)
											if np.mean(self.assesser)<avecorthresh:
												stop_training = True

								ep_steps = 0
								num_correct = 0
								prevenvtype = self.env.envtype
								self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',set_start_state=s_cur,training_deck_indices=training_deck_indices)

				elif TO_TRAIN=='DNET': #if getting experience for the decision network training or the newexpert n3
					dnet_ch_state_in = self.working_copy_network.dnet_lstm_state_init #defines ch_state_in as zeros
					self.dnet_train_rnn_state = dnet_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
					n1_ch_state_in = self.working_copy_network.n1_lstm_state_init #defines ch_state_in as zeros
					n2_ch_state_in = self.working_copy_network.n2_lstm_state_init #defines ch_state_in as zeros
					n3_ch_state_in = self.working_copy_network.n3_lstm_state_init

					ep_steps = 0
					num_correct = 0
					if on_ep==True:
						ep_term = False
						wholess = 0
						thanwho = 3
					else:
						wholess = 0 #cardpulls
						thanwho = 10 #cards_to_train_on
					while wholess < thanwho:
						dnet_policy_rec, dnet_ch_state_out, lv = sess.run([self.working_copy_network.dnet_policy_layer_output,
							self.working_copy_network.dnet_state_out,
							self.working_copy_network.lv],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.dnet_state_in[0]:dnet_ch_state_in[0],
							self.working_copy_network.dnet_state_in[1]:dnet_ch_state_in[1]})

						#get the expert networks (1 & 2) policy rec and LSTM states to continue passing
						n1pr, n2pr, n3pr, n1_ch_state_out, n2_ch_state_out, n3_ch_state_out = sess.run([self.working_copy_network.n1_policy_layer_output,
							self.working_copy_network.n2_policy_layer_output,
							self.working_copy_network.n3_policy_layer_output,
							self.working_copy_network.n1_state_out,
							self.working_copy_network.n2_state_out,
							self.working_copy_network.n3_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
							self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1],
							self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
							self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1],
							self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
							self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
						
						#put all the expert network policy recommendations into a list to choose from
						pr_list = [n1pr,n2pr,n3pr]

						if WHICH_DNET=='1e':
							dnet_policy_rec[0][1] = 0 #if not using n3, don't let choose n3
							dnet_policy_rec[0][2] = 0 #if not using n3, don't let choose n3
						elif WHICH_DNET=='2e':
							dnet_policy_rec[0][2] = 0 #if not using n3, don't let choose n3

						#decide which expert network to use based on dnet's policy_rec
						d_cur = worker_choose_action(dnet_policy_rec) #d_cur is the 1-hot action vector that is the length of the number of expert networks
						a_cur = worker_decide(d_cur,pr_list) #choses which policy to use and then uses that policy to select action
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, dnet_ch_state_in, a_cur, r_cur, s_new, ep_term, d_cur]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						dnet_ch_state_in = dnet_ch_state_out
						n1_ch_state_in = n1_ch_state_out
						n2_ch_state_in = n2_ch_state_out
						n3_ch_state_in = n3_ch_state_out
						ep_steps += 1
						num_correct += correct_ind
						if (on_ep==True) & (ep_term==True):
							wholess += 1 #if on_ep=True then wholess is indicator of when ep is over -> train
						elif on_ep==False:
							wholess += 1 #if on_ep=False then wholess is counting cardpulls

						#check if max episode length has been reached
						if wholess == thanwho:
							value_pred = sess.run([self.working_copy_network.dnet_value_layer_output],
								feed_dict={self.working_copy_network.inputs:s_cur,
								self.working_copy_network.dnet_state_in[0]:dnet_ch_state_in[0],
								self.working_copy_network.dnet_state_in[1]:dnet_ch_state_in[1]})
							bootstrap_value = value_pred #this is scalar passed to train() below
						
						if ep_term==True:
							if self.name=='workerd0':
								if train_to_profst==True: #if training until proficient or stable
									last400ep = np.append(last400ep,ep_steps)
									if len(last400ep)>=10:
										last10ep = last400ep[-10:] #get last 10 episodes
										if len(np.where(last10ep<15)[0])==10: #if last 10 episodes were less than 15 steps, proficiency achieved: stop training
											stop_training = True
									if len(last400ep) > 400: #if >400 points can test for stability criteria
										last400ep = last400ep[-400:] #pop off first one so it is len=400
										if np.abs(np.mean(last400ep[-200:])-np.mean(last400ep[0:200]))<1: #if diff between mean of last 200 and previous 200 is <1 stability achieved: stop training
											stop_training = True

								if train_assesser_on==True:
									self.assesser.append(ep_steps)
									if len(self.assesser)>aveover:
										self.assesser.pop(0)
										if np.mean(self.assesser)<avecorthresh:
											stop_training = True

							ep_steps = 0
							num_correct = 0
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',set_start_state=s_cur,training_deck_indices=training_deck_indices)

				elif TO_TRAIN=='DNETAllExpert':
					dnet_ch_state_in = self.working_copy_network.dnet_lstm_state_init #defines ch_state_in as zeros
					n1_ch_state_in = self.working_copy_network.n1_lstm_state_init #defines ch_state_in as zeros
					n2_ch_state_in = self.working_copy_network.n2_lstm_state_init #defines ch_state_in as zeros
					n3_ch_state_in = self.working_copy_network.n3_lstm_state_init #defines ch_state_in as zeros
					self.n1_train_rnn_state = n1_ch_state_in
					self.n2_train_rnn_state = n2_ch_state_in
					self.n3_train_rnn_state = n3_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
					self.dnet_train_rnn_state = dnet_ch_state_in

					ep_steps = 0
					num_correct = 0
					if on_ep==True:
						ep_term = False
						wholess = 0
						thanwho = 3
					else:
						wholess = 0 #cardpulls
						thanwho = 10 #cards_to_train_on
					while wholess < thanwho:
						dnet_policy_rec, dnet_ch_state_out = sess.run([self.working_copy_network.dnet_policy_layer_output,
							self.working_copy_network.dnet_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.dnet_state_in[0]:dnet_ch_state_in[0],
							self.working_copy_network.dnet_state_in[1]:dnet_ch_state_in[1]})

						#get the expert networks (1 & 2) policy rec and LSTM states to continue passing
						n1pr, n2pr, n3pr, n1_ch_state_out, n2_ch_state_out, n3_ch_state_out = sess.run([self.working_copy_network.n1_policy_layer_output,
							self.working_copy_network.n2_policy_layer_output,
							self.working_copy_network.n3_policy_layer_output,
							self.working_copy_network.n1_state_out,
							self.working_copy_network.n2_state_out,
							self.working_copy_network.n3_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
							self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1],
							self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
							self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1],
							self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
							self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
						
						#put all the expert network policy recommendations into a list to choose from
						pr_list = [n1pr,n2pr,n3pr]

						if WHICH_DNET=='1e':
							dnet_policy_rec[0][1] = 0 #if not using n3, don't let choose n3
							dnet_policy_rec[0][2] = 0 #if not using n3, don't let choose n3
						elif WHICH_DNET=='2e':
							dnet_policy_rec[0][2] = 0 #if not using n3, don't let choose n3

						#decide which expert network to use based on dnet's policy_rec
						d_cur = worker_choose_action(dnet_policy_rec) #d_cur is the 1-hot action vector that is the length of the number of expert networks
						a_cur = worker_decide(d_cur,pr_list) #choses which policy to use and then uses that policy to select action
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, n3_ch_state_in, a_cur, r_cur, s_new, ep_term, d_cur]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						dnet_ch_state_in = dnet_ch_state_out
						n1_ch_state_in = n1_ch_state_out
						n2_ch_state_in = n2_ch_state_out
						n3_ch_state_in = n3_ch_state_out
						ep_steps += 1
						num_correct += correct_ind #will add 1 if was correct, 0 if incorrect
						if (on_ep==True) & (ep_term==True):
							wholess += 1 #if on_ep=True then wholess is indicator of when ep is over -> train
						elif on_ep==False:
							wholess += 1 #if on_ep=False then wholess is counting cardpulls

						#check if max episode length has been reached
						if wholess == thanwho:
							value_pred = sess.run([self.working_copy_network.dnet_value_layer_output],
								feed_dict={self.working_copy_network.inputs:s_cur,
								self.working_copy_network.dnet_state_in[0]:dnet_ch_state_in[0],
								self.working_copy_network.dnet_state_in[1]:dnet_ch_state_in[1]})
							bootstrap_value = value_pred #this is scalar passed to train() below
						
						if ep_term==True:
							if self.name=='workerdne0':
								if train_to_profst==True: #if training until proficient or stable
									last400ep = np.append(last400ep,ep_steps)
									if len(last400ep)>=10:
										last10ep = last400ep[-10:] #get last 10 episodes
										if len(np.where(last10ep<15)[0])==10: #if last 10 episodes were less than 15 steps, proficiency achieved: stop training
											stop_training = True
									if len(last400ep) > 400: #if >400 points can test for stability criteria
										last400ep = last400ep[-400:] #pop off first one so it is len=400
										if np.abs(np.mean(last400ep[-200:])-np.mean(last400ep[0:200]))<1: #if diff between mean of last 200 and previous 200 is <1 stability achieved: stop training
											stop_training = True

								if train_assesser_on==True:
									self.assesser.append(ep_steps)
									if len(self.assesser)>aveover:
										self.assesser.pop(0)
										if np.mean(self.assesser)<avecorthresh:
											stop_training = True

							ep_steps = 0
							num_correct = 0
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='train',set_start_state=s_cur,training_deck_indices=training_deck_indices) #for each ep make a new env obj to get new baseline probs of A and B

				if stop_training==True:
					coord.request_stop()

				self.train(training_data,bootstrap_value,GAMMA,sess)

				GLOBAL_EPISODE_COUNTER += 1
				if GLOBAL_EPISODE_COUNTER >= EPS_TO_TRAIN_ON:
					coord.request_stop()

				#while using_workers_to_optimize is still True, go back to start, i.e. start a new episode
				#go get copy of most uptodate central_network (post applying the latest gradient update) and have at it again

	def test(self,sess,testdatapath,getnumep):
		global test_who, test_onwhom, use_random_e, NUM_TRAIN_EPS, GLOBAL_EPISODE_COUNTER
		fileroot = testdatapath + '_' + test_who + '_ON_env' + str(test_onwhom) #this creates a string of the form
		print ("Starting " + self.name + " for testing")
		with sess.as_default(), sess.graph.as_default():
			sess.run(resetzero_network_vars('central_network','central_network'))
			sess.run(self.working_copy_network_params)
			ep_num=0
			firstpriorenvtype = True
			while (ep_num < getnumep):
				if ep_num % 100 == 0: print('starting expert test episode ' + str(ep_num))
				#begin episode
				training_data = []

				#if its the first time, get a random envtype to be the t-1 envtype; otherwise give it the last envtype so it picks a different one
				if firstpriorenvtype==True:
					prevenvtype = np.random.choice(np.array([0,1,2]),p=[1/3,1/3,1/3])
					firstpriorenvtype = False
				else:
					prevenvtype = self.env.envtype

				self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='test',training_deck_indices=training_deck_indices)
				start_state = We.get_start_state_from_env(self.env)
				s_cur = start_state

				ep_steps = 0

				if test_who == 'n1':
					n1_ch_state_in = self.working_copy_network.n1_lstm_state_init #defines ch_state_in as zeros
					self.n1_train_rnn_state = n1_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out

					test_trial_num = 0
					trial_to_train_on = getnumep
					num_correct = 0
					while test_trial_num < trial_to_train_on:
						#keep track of [s_cur,a_cur,r_cur,s_new,ep_term]
						#feed st to network to get policy and value output
						#policy_rec, value_pred, ch_state_out = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)
						n1_policy_rec, n1_ch_state_out = sess.run([self.working_copy_network.n1_policy_layer_output,
							self.working_copy_network.n1_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
							self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1]})
						#choose action based on policy_rec
						a_cur = worker_choose_action(n1_policy_rec) #a_cur is the 1-hot action vector
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, n1_ch_state_in, a_cur, r_cur, s_new, ep_term]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						n1_ch_state_in = n1_ch_state_out
						ep_steps += 1
						num_correct += correct_ind

						#check if max episode length has been reached
						if ep_term == True:
							array_training_data = np.array(training_data)
							stacked_action = np.vstack(array_training_data[:,2])
							ad = np.sum(stacked_action,axis=0)

							ep_steps = 0
							num_correct = 0
							test_trial_num += 1
							ep_num += 1
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='test',set_start_state=s_cur,training_deck_indices=training_deck_indices)

				elif test_who=='n2':
					n2_ch_state_in = self.working_copy_network.n2_lstm_state_init #defines ch_state_in as zeros
					self.n2_train_rnn_state = n2_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out

					test_trial_num = 0
					trial_to_train_on = getnumep
					num_correct = 0
					while test_trial_num < trial_to_train_on:
						n2_policy_rec, n2_ch_state_out = sess.run([self.working_copy_network.n2_policy_layer_output,
							self.working_copy_network.n2_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
							self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1]})
						#choose action based on policy_rec
						a_cur = worker_choose_action(n2_policy_rec) #a_cur is the 1-hot action vector
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, n2_ch_state_in, a_cur, r_cur, s_new, ep_term]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						n2_ch_state_in = n2_ch_state_out
						ep_steps += 1
						num_correct += correct_ind

						#check if max episode length has been reached
						if ep_term == True:
							array_training_data = np.array(training_data)
							stacked_action = np.vstack(array_training_data[:,2])
							ad = np.sum(stacked_action,axis=0)

							ep_steps = 0
							num_correct = 0
							test_trial_num += 1
							ep_num += 1
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='test',set_start_state=s_cur,training_deck_indices=training_deck_indices)

				elif test_who=='n3':
					n3_ch_state_in = self.working_copy_network.n3_lstm_state_init #defines ch_state_in as zeros
					self.n3_train_rnn_state = n3_ch_state_in #to do training need to create this self variable to pass the LSTM state in and out

					test_trial_num = 0
					trial_to_train_on = getnumep
					num_correct = 0
					while test_trial_num < trial_to_train_on:
						n3_policy_rec, n3_ch_state_out = sess.run([self.working_copy_network.n3_policy_layer_output,
							self.working_copy_network.n3_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
							self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
						#choose action based on policy_rec
						a_cur = worker_choose_action(n3_policy_rec) #a_cur is the 1-hot action vector
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, n3_ch_state_in, a_cur, r_cur, s_new, ep_term]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						n3_ch_state_in = n3_ch_state_out
						ep_steps += 1
						num_correct += correct_ind

						#check if max episode length has been reached
						if ep_term == True:
							array_training_data = np.array(training_data)
							stacked_action = np.vstack(array_training_data[:,2])
							ad = np.sum(stacked_action,axis=0)
							
							ep_steps = 0
							num_correct = 0
							test_trial_num += 1
							ep_num += 1
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='test',set_start_state=s_cur,training_deck_indices=training_deck_indices)

				elif test_who == 'dnet':
					dnet_ch_state_in = self.working_copy_network.dnet_lstm_state_init #defines ch_state_in as zeros
					n1_ch_state_in = self.working_copy_network.n1_lstm_state_init #defines ch_state_in as zeros
					n2_ch_state_in = self.working_copy_network.n2_lstm_state_init #defines ch_state_in as zeros
					n3_ch_state_in = self.working_copy_network.n3_lstm_state_init #defines ch_state_in as zeros

					test_trial_num = 0
					trial_to_train_on = getnumep
					num_correct = 0
					while test_trial_num < trial_to_train_on:
						dnet_policy_rec, dnet_ch_state_out = sess.run([self.working_copy_network.dnet_policy_layer_output,
							self.working_copy_network.dnet_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.dnet_state_in[0]:dnet_ch_state_in[0],
							self.working_copy_network.dnet_state_in[1]:dnet_ch_state_in[1]})

						#get the expert networks (1 & 2) policy rec and LSTM states to continue passing
						n1pr, n2pr, n3pr, n1_ch_state_out, n2_ch_state_out, n3_ch_state_out = sess.run([self.working_copy_network.n1_policy_layer_output,
							self.working_copy_network.n2_policy_layer_output,
							self.working_copy_network.n3_policy_layer_output,
							self.working_copy_network.n1_state_out,
							self.working_copy_network.n2_state_out,
							self.working_copy_network.n3_state_out],
							feed_dict={self.working_copy_network.inputs:s_cur,
							self.working_copy_network.n1_state_in[0]:n1_ch_state_in[0],
							self.working_copy_network.n1_state_in[1]:n1_ch_state_in[1],
							self.working_copy_network.n2_state_in[0]:n2_ch_state_in[0],
							self.working_copy_network.n2_state_in[1]:n2_ch_state_in[1],
							self.working_copy_network.n3_state_in[0]:n3_ch_state_in[0],
							self.working_copy_network.n3_state_in[1]:n3_ch_state_in[1]})
						
						#put all the expert network policy recommendations into a list to choose from
						pr_list = [n1pr,n2pr,n3pr]

						#decide which expert network to use based on dnet's policy_rec
						if WHICH_DNET=='2e': #only use n1 and n2
							if use_random_e == True:
								setds = [np.array([1,0,0]),np.array([0,1,0])]
								d_cur = setds[np.random.choice([0,1],p=(0.5,0.5))]
							else:
								dnet_policy_rec[0][2] = 0 #if not using n3, don't let choose n3
								d_cur = worker_choose_action(dnet_policy_rec) #d_cur is the 1-hot action vector that is the length of the number of expert networks
						elif WHICH_DNET=='1e': #shouldn't do this; instead test n1
							dnet_policy_rec[0][1] = 0
							dnet_policy_rec[0][2] = 0
							d_cur = worker_choose_action(dnet_policy_rec)
						else:
							if use_random_e == True:
								setds = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
								d_cur = setds[np.random.choice([0,1,2],p=(1/3,1/3,1/3))]
							else:
								d_cur = worker_choose_action(dnet_policy_rec)
						a_cur = worker_decide(d_cur,pr_list) #choses which policy to use and then uses that policy to select action
						r_cur, s_new, ep_term, correct_ind, action_t = worker_act(self.env,s_cur,a_cur)
						#now for this step we have [s_cur, ch_state_in, a_cur, r_cur, s_new, ep_term] to use for training

						new_step_in_environment = [s_cur, dnet_ch_state_in, d_cur, r_cur, s_new, ep_term, np.where(a_cur==1)[0]]
						training_data.append(new_step_in_environment) #this is the data to calculate gradients with

						s_cur = s_new
						dnet_ch_state_in = dnet_ch_state_out
						n1_ch_state_in = n1_ch_state_out
						n2_ch_state_in = n2_ch_state_out
						n3_ch_state_in = n3_ch_state_out
						ep_steps += 1
						num_correct += correct_ind

						if ep_term == True:
							# self.assesser.append(ep_steps) #keep track of all episodes epl

							array_training_data = np.array(training_data)
							stacked_decision = np.vstack(array_training_data[:,2])
							stacked_states = np.vstack(array_training_data[:,0])
							stacked_actions = np.vstack(array_training_data[:,6])
							stacked_rewards = np.vstack(array_training_data[:,3])

							dd = np.sum(stacked_decision,axis=0)

							ep_steps = 0
							num_correct = 0
							test_trial_num += 1
							ep_num += 1
							prevenvtype = self.env.envtype
							self.env = We.make_new_env(last_ep_type=prevenvtype,actorenv=self.actorenv,status='test',set_start_state=s_cur,training_deck_indices=training_deck_indices)

			print('testing complete')

##################################################################################################################################
####################################################################MAIN##########################################################

#main
tf.reset_default_graph()

#sys.argv[1] - first arg is the decision network size 98
#sys.argv[2] - second arg is the expert network size: 19
#sys.argv[3] - third arg is the env_train_on: 0, 1, 2, None
if str(sys.argv[3])=='None':
	TRAIN_ENV = None
else:
	TRAIN_ENV = int(sys.argv[3])
#sys.argv[4] - fourth arg is the #ep to train on, e.g. 3000
#sys.argv[5] - fifth arg is gpu to use (if multiple to choose from): 0
#sys.argv[6] - sixth arg is lesion type: 0 (no lesion), 1-12
#sys.argv[7] - seventh arg is degree of ablation or knockdown in lesions: [0.0-1.0]
#sys.argv[8] - eighth arg is carddeck: 0 is fulldeck; 1 is MWCST deck (no ambiguous cards)
#sys.argv[9] - ninth arg is run number: 0
#
#	python3 DynaMoE_lesion.py [NETSZ_D] [NETSZ_E] [TRAIN_ENV] [EPS_TO_TRAIN_ON] [GPU] [LTYPE] [P_ABL] [CARDDECK] [RUNNO]

LTYPE = int(sys.argv[6]) #Lesion type: 0=no_lesion,
	#impaired input to gate: 1=gate_no_rt-1, 2=gate_no_at-1, 3=gate_no_art-1
	#impaired gate output: 4=gate_output
	#impaired gate output synapses: 5=w_hv, 6=w_hpi
	#impaired value connections: 6,10=v_mask
	#impaired policy connections: 8,10=pi_mask
	#impaired network dynamics (forget gate connections ablation): 12=forget_gate
P_ABL = float(sys.argv[7]) #degree of ablation or knockdown
RUNNO = int(sys.argv[9])

GLOBAL_EPISODE_COUNTER = 0
TRAIN_EP_COUNT = 0
EPS_TO_TRAIN_ON = int(sys.argv[4])
MAX_EPISODE_LEN = 10000 #how many time steps is maximum length ep; this may depend on environment
TRAINER = tf.train.AdamOptimizer(learning_rate=1e-3)
GAMMA = 0
NUMBER_OF_WORKERS = 12
RNDMSD = None
useFD = int(sys.argv[8]) #CARDDECK -- 0 is fulldeck; 1 is MWCST deck (no ambiguous cards)
NUM_TRAIN_EPS = 0
dw_env_train = [TRAIN_ENV]

train_assesser_on = True
avecorthresh = 4 #4; ave performance threshold to assess for stopping training
aveover = 100 #100; number of ep to ave over to assess
train_to_profst = False #setting that stops training if "stable"; checks if difference in ave performance over last 2 blocks of 200 ep is <1 (considered stable)

subfldr = 'LESION' #'n123' or 'OI' or 'OQWK'; change depending on mode being used in

NETSZ_D = int(sys.argv[1]) #size of gating network
NETSZ_E = int(sys.argv[2]) #size of expert networks

STATE_SPACE = We.get_state_space() #size of state space
ACTION_SPACE = We.get_action_space() #size of action space

if LTYPE==4: #ablate some percentage of output cells
	rnnout_Lm = np.random.choice(2,NETSZ_D,p=[P_ABL, (1-P_ABL)]) #p=[0.1,0.9] means 10% chance a given unit's output to policy and value layers is ablated; ablation mask set and held constant from beginning
elif LTYPE==5:
	rnnout_Lm = np.ones(NETSZ_D) * P_ABL #amount output is muted
else:
	rnnout_Lm = np.ones(NETSZ_D) #no change to output units

if (LTYPE==6) | (LTYPE==10): 
	v_mask = np.random.choice(2,[NETSZ_D,1],p=[P_ABL,(1-P_ABL)]) #P_ABL likelihood a given connection from output to pi layer is dropped
elif (LTYPE==7) | (LTYPE==11):
	v_mask = np.ones([NETSZ_D,1]) * P_ABL
else:
	v_mask = np.ones([NETSZ_D,1])

if (LTYPE==8) | (LTYPE==10):
	pi_mask = np.random.choice(2,[NETSZ_D,3],p=[P_ABL,(1-P_ABL)])
elif (LTYPE==9) | (LTYPE==11):
	pi_mask = np.ones([NETSZ_D,3]) * P_ABL
else:
	pi_mask = np.ones([NETSZ_D,3])

if LTYPE==12: #ablate connections between input and forget gate
	input_w_mask = np.concatenate((np.ones([STATE_SPACE+NETSZ_D,NETSZ_D*2]),np.random.choice(2,[STATE_SPACE+NETSZ_D,NETSZ_D],p=[P_ABL,(1-P_ABL)]),np.ones([STATE_SPACE+NETSZ_D,NETSZ_D])),axis=1)
else:
	input_w_mask = np.ones([STATE_SPACE+NETSZ_D,NETSZ_D*4])
	
cchars = [0,1,2,3]
fdeck = list(itertools.product(cchars,repeat=3))
mdeck = list(itertools.permutations(cchars,3))
if useFD==0:
	if RNDMSD==None:
		training_deck_indices = [item for item in range(len(fdeck))]
# 	else: #if wanted to train with part of deck and test on rest of deck -- if uncomment, then need to uncomment corresponding in WCST_env_inarow.py file
# 		np.random.seed(RNDMSD)
# 		training_deck_indices = [item for item in np.random.choice(range(len(fdeck)),round(len(fdeck)*2/3),replace=False)] #list of which cards to use for training
# 		np.random.seed(None)
	else:
		print('Error: Not configured for split train-testing deck; please uncomment relevant areas')
else:
	training_deck_indices = [item for item in range(len(mdeck))]

#create central_network
central_network = the_network(state_space=STATE_SPACE, action_space=ACTION_SPACE, name='central_network', trainer=None)

# create worker objects
#workers to train the old experts, n1 and n2
n1workers = []
for i in range(NUMBER_OF_WORKERS):
	n1workers.append(worker(name='n1'+str(i),trainer=TRAINER,actorenv=[0])) #half the actors to train on shape-matching
n2workers = []
for i in range(NUMBER_OF_WORKERS):
	n2workers.append(worker(name='n2'+str(i),trainer=TRAINER,actorenv=[1])) #half the actors to train on color-matching
n3workers = []
for i in range(NUMBER_OF_WORKERS):
	n3workers.append(worker(name='n3'+str(i),trainer=TRAINER,actorenv=[2])) #half the actors to train on color-matching

#workers to train the decision network DNET
dworkers = []
for i in range(NUMBER_OF_WORKERS):
	dworkers.append(worker(name='d'+str(i),trainer=TRAINER,actorenv=dw_env_train))

dneworkers = []
for i in range(NUMBER_OF_WORKERS):
	dneworkers.append(worker(name='dne'+str(i),trainer=TRAINER,actorenv=dw_env_train))


#Tell what to do with network - save, restore, train, test------------------------------------------------------------------
#saver
filerootpath = "/home/outputfiles/"
saver = tf.train.Saver()
dosave = False #set to True to save; False to not save
savepath = filerootpath+"models/savedmodel.ckpt"
dorestore = False
restorepath = filerootpath+"models/savedmodel.ckpt" #Set to restore path if want to restore
trainnetwork = True #set to True if want to train network; False if just restoring and network and testing it
testnetwork = True #set to True if want to do test run of network
traindatapath = filerootpath+"data/train_"
testdatapath = filerootpath+"data/test_" #give path and prefix for data to save
test_who = 'dnet' #network to test: n1, n2, n3 (experts), dnet (gating)
test_onwhom = [None] #environment type to test on: [0], [1], [2], [None] (random), or random choice from 2 envs e.g. [0,1]
use_random_e = False #option to use random expert instead of letting gating network decide
getnumep = 1000 #number of episodes to test on
TO_TRAIN = 'DNET' #Experts_n1, Experts_n2, Experts_n3, DNET, DNETAllExpert
TrainNewOnly = True #option to train all experts simultaneous or only the newly added expert network
WHICH_DNET = '3e' # '2e' if training with just n1 and n2; '3e' if training with all 3 experts

the_test_worker = worker(name='_testman',trainer=TRAINER,actorenv=test_onwhom)

#---------------------------------------------------------------------------------------------------------------------------

# create the tf.Session
UGPU = str(sys.argv[5])
with tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(UGPU, .2))) as sess:
	#load(aka restore) network or initialize new one
	if dorestore == True:
		saver.restore(sess,restorepath)
		print("Model restored: " + restorepath)
	elif dorestore == False:
		sess.run(tf.global_variables_initializer()) #initialize variables

	sess.run(resetzero_network_vars('central_network','central_network'))

	#if training network, train using workers
	if trainnetwork == True:
		coord = tf.train.Coordinator()
		#start each worker in a separate thread
		if TO_TRAIN=='Experts_n1':
			n1_worker_threads = [] #list of worker threads
			for worker in n1workers: #for each worker object in the workers list of them
				worker_experience = lambda: worker.get_experience(sess=sess,coord=coord)
				newthread = threading.Thread(target=(worker_experience))
				newthread.start() #start the worker
				time.sleep(0.5) #wait a bit
				n1_worker_threads.append(newthread) #add the started worker to the list of worker_threads
			coord.join(n1_worker_threads) #pass the worker threads to the coordinator that will apply join to them
										#i.e. wait for them to finish
		elif TO_TRAIN=='Experts_n2':
			n2_worker_threads = [] #list of worker threads
			for worker in n2workers: #for each worker object in the workers list of them
				worker_experience = lambda: worker.get_experience(sess=sess,coord=coord)
				newthread = threading.Thread(target=(worker_experience))
				newthread.start() #start the worker
				time.sleep(0.5) #wait a bit
				n2_worker_threads.append(newthread) #add the started worker to the list of worker_threads
			coord.join(n2_worker_threads) #pass the worker threads to the coordinator that will apply join to them
										#i.e. wait for them to finish
		elif TO_TRAIN=='Experts_n3':
			n3_worker_threads = [] #list of worker threads
			for worker in n3workers: #for each worker object in the workers list of them
				worker_experience = lambda: worker.get_experience(sess=sess,coord=coord)
				newthread = threading.Thread(target=(worker_experience))
				newthread.start() #start the worker
				time.sleep(0.5) #wait a bit
				n3_worker_threads.append(newthread) #add the started worker to the list of worker_threads
			coord.join(n3_worker_threads) #pass the worker threads to the coordinator that will apply join to them
										#i.e. wait for them to finish
		elif TO_TRAIN=='DNET':
			dworker_threads = []
			for worker in dworkers: #for each worker object in the workers list of them
				worker_experience = lambda: worker.get_experience(sess=sess,coord=coord)
				newthread = threading.Thread(target=(worker_experience))
				newthread.start() #start the worker
				time.sleep(0.5) #wait a bit
				dworker_threads.append(newthread) #add the started worker to the list of worker_threads
			coord.join(dworker_threads) #pass the worker threads to the coordinator that will apply join to them
										#i.e. wait for them to finish
		elif TO_TRAIN=='DNETAllExpert':
			dneworker_threads = []
			for worker in dneworkers: #for each worker object in the workers list of them
				worker_experience = lambda: worker.get_experience(sess=sess,coord=coord)
				newthread = threading.Thread(target=(worker_experience))
				newthread.start() #start the worker
				time.sleep(0.5) #wait a bit
				dneworker_threads.append(newthread) #add the started worker to the list of worker_threads
			coord.join(dneworker_threads) #pass the worker threads to the coordinator that will apply join to them
										#i.e. wait for them to finish

		#if saving network, save
		if dosave == True:
			save_path = saver.save(sess, savepath)
			print("saved to " + savepath)

		#once the network is trained: create a new worker to test it out
		if testnetwork == True:
			the_test_worker.test(sess=sess,testdatapath=testdatapath,getnumep=getnumep)




