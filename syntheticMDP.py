# Synthetic Experiment on MDP Value Iteration algorithms
import random
import numpy as np 
import copy
import time
import statistics
import matplotlib.pyplot as plt

# Large action space: reduce variation in num of iterations
class MDP:
    def __init__(self, NUM_STATES = 10, NUM_ACTIONS = 50, \
            lb = 30, ub = 50, GAMMA = 0.9, TOL = 1e-7, transitionError=False, rewardError=False):
        # Generates a random int vector with size = NUM_STATES
        ## [ |A1|, |A2|, ..., |Am| ]
        self.NUM_STATES=NUM_STATES
        self.NUM_ACTIONS=NUM_ACTIONS
        self.states = range(NUM_STATES)
        self.actions = range(NUM_ACTIONS)
        self.lb = lb
        self.up = ub
        self.GAMMA = GAMMA
        self.TOL = TOL
        
        self.stateActionSize = np.random.randint(lb, ub + 1,  size=NUM_STATES) 
        self.stateActions = {}
        for i in range(NUM_STATES):
            self.stateActions[i] = sorted(random.sample(self.actions, self.stateActionSize[i]))
        
        self.stateActionCounts = {}
        for i in range(NUM_STATES):
            self.stateActionCounts[i] = {}
            for action in self.stateActions[i]:
                self.stateActionCounts[i][action] = 0
        
        # Generate transition probability matrix
        self.T = {}
        for i,actionSize in enumerate(self.stateActionSize):
            # for state i, pmatrix should have column size of actionSize, |Ai|
            self.T[i] = {}
            for action in self.stateActions[i]:
                # Generate probability for m states
                # assign probabilities
                p = np.random.choice([0.1, 0.1, 0.1, 0.1, 200.0], NUM_STATES) 
                # normalize
                p /= p.sum() 

                self.T[i][action] = p
        
        # reward matrix
        self.reward = {}
        for i in range(NUM_STATES):
            self.reward[i] = {}
            for action in self.stateActions[i]:
                self.reward[i][action] = np.random.rand()*2
    
    def transitionCorrupted(self, error_type="uniform"):
    	for i,actionSize in enumerate(self.stateActionSize):
    		eps = None
    		for action in self.stateActions[i]:
    			# error type
    			if error_type == "uniform":
    				eps = np.random.uniform(low=0.0, high=1.0, size=len(self.T[i][action]))
    			elif error_type == "gaussian":
    				eps = np.maximum(np.random.randn(len(self.T[i][action])), 0)
    			
    			new_p = self.T[i][action] + eps
    			new_p /= new_p.sum()
    			self.T[i][action] = new_p

    def rewardCorrupted(self, error_type="uniform"):
    	for i,actionSize in enumerate(self.stateActionSize):
    		eps = None
    		for action in self.stateActions[i]:
    			# error type
    			if error_type == "uniform":
    				eps = np.random.uniform(low=0.0, high=1.0, size=len(self.T[i][action]))
    			elif error_type == "gaussian":
    				eps = np.maximum(np.random.randn(len(self.T[i][action])), 0)

    			self.reward[i][action] += eps

def VanillaVI(mdp):
	start_time = time.time() # start timer
	
	# iteration number
	t = 0

	# value function initialization
	V = np.zeros(mdp.NUM_STATES) 
	
	policy = {i:"None" for i in range(mdp.NUM_STATES)}

	while(True):
		updated_V = copy.deepcopy(V)
		t += 1
		for i in range(mdp.NUM_STATES):
			candidates = []
			for action in mdp.stateActions[i]:
				candidate = mdp.reward[i][action] + mdp.GAMMA * (np.dot(mdp.T[i][action], V))
				candidates.append((candidate, action))
			updated_V[i] = max(candidates)[0]
			policy[i] = max(candidates)[1]

		checker = max([ abs(V[i] - updated_V[i]) for i in range(mdp.NUM_STATES) ])

		if checker <= mdp.TOL:
			execution_time = time.time() - start_time # end timer once VI finishes
			return updated_V, policy, t, execution_time
		V = updated_V

def evaluatePolicy(policy, mdp):
	# value function initialization
	V = np.zeros(mdp.NUM_STATES) 
	# iteration number
	t = 0 

	for i in range(mdp.NUM_STATES):
		action = policy[i]
		updated_V[i] = mpd.reward[i][action] + mdp.GAMMA * (np.dot(mdp.T[i][action], V))

	checker = max( [ abs(V[i] - updated_V[i]) for i in range(mdp.NUM_STATES) ])

	if checker <= mdp.TOL:
		return updated_V
	V = updated_V

def plot(x, VI, cVI):
	plt.figure()
	plt.plot(x, VI, linestyle="-", label="OriginalVI")
	plt.plot(x, cVI, linestyle="-", color="r", label="transitionCorrupted_VI")

	file_name = "computationTime.png"
	plt.grid()
	plt.xlabel("Number of States")
	plt.ylabel("Execution Time (s)")
	plt.legend(loc='upper left')
	plt.savefig(file_name)
	plt.show()

if __name__=="__main__":
	#### Setting
	# m states
	NUM_STATES = 10
	# n actions
	NUM_ACTIONS = 50
	states = range(NUM_STATES)
	actions = range(NUM_ACTIONS)
	lb = 30 # lower bound
	ub = 50 # upper bound
	GAMMA = 0.9
	TOL = 1e-7
	NUM_TRIALS=30
	K = 5

	# data for plot on computation time
	x = [] 
	VI = [] 
	cVI = []
	policyVectors = {}
	for numStates in range(10, 20, 2):
		NUM_STATES = numStates
		x.append(numStates)
		vanilaMDP = MDP(NUM_STATES=NUM_STATES, NUM_ACTIONS=NUM_ACTIONS,\
		        lb=lb, ub=ub, GAMMA=GAMMA, TOL=TOL)
		updated_V, policy, t, execution_time = VanillaVI(vanilaMDP)
		VI.append(execution_time)
		print("vanilaMDP Case, States", ":", x[-1], "Completed")
		print("Policy evaulation result :", updated_V)

		T_corruptMDP = copy.deepcopy(vanilaMDP)
		T_corruptMDP.transitionCorrupted("gaussian")

		updated_V2, policy2, t, execution_time = VanillaVI(T_corruptMDP)
		cVI.append(execution_time)
		
		print("corrupted MDP Case, States", ":", x[-1], "Completed")
		print("Policy evaulation result :", updated_V2)

		policyVectors["vanilaMDP"] = policy
		policyVectors["T_corruptMDP"] = policy2
	
	print("Policy on original MDP setting : \n", policyVectors["vanilaMDP"])
	print("Policy on transition corrupted MDP setting : \n", policyVectors["T_corruptMDP"])
	plot(x, VI, cVI)



