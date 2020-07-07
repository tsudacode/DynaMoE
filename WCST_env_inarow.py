#Environment for Wisconsin card task (WIC)

import numpy as np
import itertools

class WIC_env():
	def __init__(self, last_ep_type=1, actorenv=[None],status='train',set_start_state=np.array([]),training_deck_indices=[0]):
		self.status=status
		self.action_space_size = 4 #action space is {a_stackA, a_stackB, a_stackC, a_stackD}
									#stackA is {TRIANGLE,RED,1} aka {0,0,0}
									#stackB is {STAR,GREEN,2} aka {1,1,1}
									#stackC is {CROSS,YELLOW,3} aka {2,2,2}
									#stackD is {CIRCLE,BLUE,4} aka {3,3,3}
		self.possible_actions = ["stackA", "stackB", "stackC", "stackD"]
		self.state_space_size = 5 #state is represented by stim card characteristic (shape, color, number) and
									#previous action (a_t-1) and reward (r_t-1)
		#for start state choose random previous action and reward and cards
		self.cardchars = [0,1,2,3]
		self.fulldeck = list(itertools.product(self.cardchars,repeat=3))

		self.training_deck_indices = training_deck_indices
		if len(training_deck_indices)==64: #if its the whole deck use it for both train and test
			self.carddeck = [tuple(item) for item in np.array(self.fulldeck)[self.training_deck_indices]]
		else:
			if self.status=='train':
				self.carddeck = [tuple(item) for item in np.array(self.fulldeck)[self.training_deck_indices]]
			elif self.status=='test':
				self.carddeck = [tuple(item) for item in np.delete(np.array(self.fulldeck),self.training_deck_indices,0)]

		randnewcard = False
		if len(set_start_state)==0:
			randnewcard=True

		if randnewcard==True: #if no start state card given, draw one
			random_pick = np.random.randint(0,len(self.carddeck))
			start_card = self.carddeck[random_pick]
			self.carddeck.pop(random_pick)

			start_c1_shape=start_card[0]
			start_c1_col=start_card[1]
			start_c1_num=start_card[2]
			start_a=np.random.choice([0,1,2,3],p=[0.25,0.25,0.25,0.25])
			start_r=np.random.choice([-5,5],p=[0.5,0.5])
			self.start_state = np.array([start_c1_shape,start_c1_col,start_c1_num,start_a,start_r],ndmin=2)
			self.state = self.start_state
		else: #pop off the start state card and set self.state to given start state
			startcard = set_start_state[0][0:3]
			startcardindex = self.carddeck.index(tuple(startcard))
			self.carddeck.pop(startcardindex) #remove the start card from deck
			self.start_state = set_start_state
			self.state = self.start_state

		if actorenv[0] != None:
			if len(actorenv)==1:
				self.envtype = actorenv[0]
			elif len(actorenv)==2: #if testing/training on 2 envs then randomly pick new one each time
				self.envtype = np.random.choice(actorenv)
		else:
			envtypes = np.array([0,1,2])
			envchoice = np.delete(envtypes,np.where(np.array(last_ep_type)==envtypes)[0][0],0)
			self.envtype = np.random.choice(envchoice,p=[0.5,0.5]) #when create environment, select what the matching rule is for this episode; must be different from most recent
								#matching rule can be shape {0}, color {1}, or number {2}

		self.correct_in_row = 0
		self.trial_number = 0
		self.ep_length = 200 #if ep isn't finished, last card is the new state for the bootstrap value estimate


	def step(self, action_command):
		correct_ind = 0
		self.trial_number += 1
		if action_command==0: #if put in stackA {TRIANGLE,RED,1} aka {0,0,0}
			if self.envtype==0: #if its shape-matching env
				if self.state[0][0]==0: #if stim card is TRIANGLE give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==1: #if its color-matching env
				if self.state[0][1]==0: #if stim card is RED give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==2: #if its number-matching env
				if self.state[0][2]==0: #if stim card is 1 give reward, else don't
					reward = 1
				else:
					reward = 0
		##################################################################################
		elif action_command==1: #if put in stackB {STAR,GREEN,2} aka {1,1,1}
			if self.envtype==0: #if its shape-matching env
				if self.state[0][0]==1: #if stim card is STAR give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==1: #if its color-matching env
				if self.state[0][1]==1: #if stim card is GREEN give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==2: #if its number-matching env
				if self.state[0][2]==1: #if stim card is 2 give reward, else don't
					reward = 1
				else:
					reward = 0
		##################################################################################
		elif action_command==2: #if put in stackC {CROSS,YELLOW,3} aka {2,2,2}
			if self.envtype==0: #if its shape-matching env
				if self.state[0][0]==2: #if stim card is CROSS give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==1: #if its color-matching env
				if self.state[0][1]==2: #if stim card is YELLOW give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==2: #if its number-matching env
				if self.state[0][2]==2: #if stim card is 3 give reward, else don't
					reward = 1
				else:
					reward = 0
		##################################################################################
		elif action_command==3: #if put in stackD {CIRCLE,BLUE,4} aka {3,3,3}
			if self.envtype==0: #if its shape-matching env
				if self.state[0][0]==3: #if stim card is CIRCLE give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==1: #if its color-matching env
				if self.state[0][1]==3: #if stim card is BLUE give reward, else don't
					reward = 1
				else:
					reward = 0
			elif self.envtype==2: #if its number-matching env
				if self.state[0][2]==3: #if stim card is 4 give reward, else don't
					reward = 1
				else:
					reward = 0
		
		#if correct add to tally of how many correct in a row; if wrong, reset self.correct_in_row to 0
		if reward==1:
			reward=5
			self.correct_in_row += 1
			correct_ind = 1
		elif reward==0: #got it wrong
			reward=-5
			self.correct_in_row = 0
			correct_ind = 0

		if self.correct_in_row == 3: #reinforcements before termination of episode
			ep_term = True
			self.correct_in_row = 0
		elif self.trial_number == self.ep_length: #if get to ep_length card pulls and still haven't gotten 3 in a row, end ep and make a new env
			ep_term = True
			self.correct_in_row = 0
			self.trial_number = 0
		else:
			ep_term = False
		
		if len(self.carddeck)==0:
			if len(self.training_deck_indices)==64: #if its the whole deck use it for both train and test
				self.carddeck = [tuple(item) for item in np.array(self.fulldeck)[self.training_deck_indices]]
			else:
				if self.status=='train':
					self.carddeck = [tuple(item) for item in np.array(self.fulldeck)[self.training_deck_indices]]
				elif self.status=='test':
					self.carddeck = [tuple(item) for item in np.delete(np.array(self.fulldeck),self.training_deck_indices,0)]
		random_pick = np.random.randint(0,len(self.carddeck))
		new_card = self.carddeck[random_pick]
		self.carddeck.pop(random_pick)
		c1_shape=new_card[0]
		c1_col=new_card[1]
		c1_num=new_card[2]
		new_state = np.array([c1_shape,c1_col,c1_num,action_command,reward],ndmin=2)
		self.state = new_state #update the environment's state; next step will use this state
		self.action_allowed = False #next step action is not allowed

		return self.state, reward, ep_term, correct_ind


def get_state_space():
	env = WIC_env()
	return(env.state_space_size)

def get_action_space():
	env = WIC_env()
	return(env.action_space_size)

def make_new_env(last_ep_type=1, actorenv=[None],status='train',set_start_state=np.array([]),training_deck_indices=[0]):
	env = WIC_env(last_ep_type=last_ep_type, actorenv=actorenv, status=status, set_start_state=set_start_state,training_deck_indices=training_deck_indices)
	return(env)

#returns array that defines the start state
def get_start_state_from_env(env):
	obs = env.start_state
	return(obs)

def do_action_in_environment(env,state,action):
	# if np.random.rand() > 0.1:
	action_command = np.argmax(action) #in WIC_env A is 0 and B is 1; action is 1-hot so A is [1,0], B is [0,1]
	# else:
	# 	action_command = np.random.choice(np.r_[0,1:(env.action_space_size-1),(env.action_space_size-1)],p=[0.25,0.25,0.25,0.25])
	new_state_raw, reward_raw, ep_term, correct_ind = env.step(action_command)
	new_state = np.resize(new_state_raw,(1,5))
	reward = np.resize(reward_raw,(1,1))
	# if new_state == ***#terminal state:
	# 	ep_term = True #terminal state
	# else: ep_term = False #not terminal state
	return reward, new_state, ep_term, correct_ind, action_command

