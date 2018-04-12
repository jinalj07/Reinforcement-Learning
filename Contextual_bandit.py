import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 2: Contextual bandit problem 
    In this problem, we will implement an AI player for contextual multi-armed bandit problem epsilon-greedy method.
    The main goal of this problem is to get familiar with a simplified problem in reinforcement learning, and how to train the model parameters on the data from a game.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#-------------------------------------------------------
class CBandit:
    '''CBandit is a Contextual Multi-armed bandit machine. The odds of the winning for each lever also depends on the context (the state) of the machine. 
        For example, the machine can have two states, say a green light on the screen, or a red light on the screen. 
        The state of the machine can be observed by the player. '''
    # ----------------------------------------------
    def __init__(self, p):
        ''' Initializing the game. 
            Inputs:
                p: the matrix of winning probabilities of each arm at each state, a numpy matrix of n_s by n. 
                    Here n is the number of arms of the bandit. n_s is the number of states of the machine
                    p[i,j] represents the winning probability of the machine at i-th state and the j-th arm is being pulled by the player.
            Outputs:
                self.p: the matrix of winning probabilities, a numpy matrix of n_s by n. 
                self.n_s: the number of states of the machine, an integer scalar.
                self.s: the current state of the machine, an integer scalar, initialized as 0.
        '''
        self.p = p
        self.n_s = p.shape[0]
        self.s = 0


    # ----------------------------------------------
    def step(self, a):
        '''
           Given an action (the id of the arm being pulled), returning the reward based upon the winning probability of the arm. 
         The winning probability depends on the current state of the machine. 
         After each step, the machine will randomly change the state with uniform probability.
            Input:
                a: the index of the lever being pulled by the agent. a is an integer scalar between 0 and n-1. 
                    n is the number of arms in the bandit.
            Output:
                r: the reward of the previous action, a float scalar. The "win" return 1., if "lose", return 0. as the reward.
                s: the new state of the machine, an integer scalar. 
        '''

        
        rand = np.random.random()
        r = 1 if (rand < self.p[self.s,a]) else 0 
        
        s = np.random.randint(0,(self.n_s))
        self.s = s

        return r, s


#-------------------------------------------------------
class Agent(object):
    '''The agent is trying to maximize the sum of rewards (payoff) in the game using epsilon-greedy method. 
       The agent will 
                (1) with a small probability (epsilon or e), randomly pull a lever with a uniform distribution on all levers (Exploration); 
                (2) with a big probability (1-e) to pull the arm with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.'''
    # ----------------------------------------------
    def __init__(self, n, n_s, e=0.1):
        ''' Initializing the agent. 
            Inputs:
                n: the number of arms of the bandit, an integer scalar. 
                n_s: the number of states of the bandit, an integer scalar. 
                e: (epsilon) the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
            Outputs:
                self.n: the number of levers, an integer scalar. 
                self.e: the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
                self.Q: the expected ratio of rewards for pulling each lever at each state, a numpy matrix of shape n_s by n. We initialize the matrix with all-zeros.
                self.c: the counts of the number of times that each lever being pulled given a certain state. a numpy matrix of shape n_s by n, initialized as all-zeros.
                
        '''

        self.n = n
        self.e = e
        self.Q = np.zeros((n_s,n),dtype = np.float)
        self.c = np.zeros((n_s,n),dtype = np.int)


   # ----------------------------------------------
    def forward(self,s):
        '''
            The policy function of the agent.
            Inputs:
                s: the current state of the machine, an integer scalar. 
            Output:
                a: the index of the lever to pull. a is an integer scalar between 0 and n-1. 
        '''

        rand = np.random.random()
        if (rand < self.e):
            a = np.random.randint(self.n)  
        
        else:
            a = np.argmax(self.Q[s])

        return a


    #-----------------------------------------------------------------
    def update(self,s,a,r):
        '''
            Updating the parameters of the agent.
            (1) increase the count of lever
            (2) update the expected reward based upon the received reward r.
            Input:
                s: the current state of the machine, an integer scalar. 
                a: the index of the arm being pulled. a is an integer scalar between 0 and n-1. 
                r: the reward returned, a float scalar. 
        '''

        self.c[s,a] += 1
        self.Q[s,a] += (1./self.c[s,a]) * (r - self.Q[s,a])



    #-----------------------------------------------------------------
    def play(self, g, n_steps=1000):
        '''
            Play the game for n_steps steps. In each step,
            (1) pull a lever and receive the reward and the state from the game
            (2) update the parameters 
            Input:
                g: the game machine, a multi-armed bandit object. 
                n_steps: number of steps to play in the game, an integer scalar. 
        '''
        s = g.s # initial state of the game

        
        for i in range(n_steps):
            a = self.forward(s)
            r,s_new = g.step(a)
            self.update(s,a,r)
            s = s_new

