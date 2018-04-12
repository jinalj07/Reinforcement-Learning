import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 1: Multi-armed bandit problem 
    In this problem, we will implement an AI player for Multi-armed bandit problem epsilon-greedy method.
    The main goal of this problem is to get familiar with a simplified problem in reinforcement learning, and how to train the model parameters on the data from a game.
'''

#-------------------------------------------------------
class Bandit:
    '''Bandit is the Multi-armed bandit machine. Instead of one  slot machine lever, you have a number of them, say three. Each lever/arm corresponds to a probability of winning. However these odds/probabilities are hidden from the players. '''
    # ----------------------------------------------
    def __init__(self, p):
        ''' Initializing the game. 
            Inputs:
                p: the vector of winning probabilities, a numpy vector of length n. 
                    Here n is the number of arms of the bandit. 
            Outputs:
                self.p: the vector of winning probabilities, a numpy vector of length n. 
        '''

        self.p = p


    # ----------------------------------------------
    def step(self, a):
        '''
           Given an action (the id of the arm being pulled), return the reward based upon the winning probability of the arm. 
            Input:
                a: the index of the lever being pulled by the agent. a is an integer scalar between 0 and n-1. 
                    n is the number of arms in the bandit.
            Output:
                r: the reward returned to the agent, a float scalar. The "win" return 1., if "lose", return 0. as the reward.
                   The winning probabilty of this step should be the same as that of the lever being pulled by the agent.
        '''

        rand = np.random.random()
        r = 1 if (rand < self.p[a]) else 0

        return r


#-------------------------------------------------------
class Agent:
    '''The agent is trying to maximize the sum of rewards (payoff) in the game using epsilon-greedy method.
       The agent will 
                (1) with a small probability (epsilon or e), randomly pull a lever with uniform distribution on all levers (Exploration); 
                (2) with a big probability (1-e) to pull the arm with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.'''
    # ----------------------------------------------
    def __init__(self, n, e=0.1):
        ''' Initializing the agent. 
            Inputs:
                n: the number of arms of the bandit, an integer scalar. 
                e: (epsilon) the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
            Outputs:
                self.n: the number of levers, an integer scalar. 
                self.e: the probability of the agent randomly pulling a lever with uniform probability. e is a float scalar between 0. and 1. 
                self.Q: the expected ratio of rewards for pulling each lever, a numpy vector of length n. We initialize the vector with all-zeros.
                self.c: the counts of the number of times that each lever being pulled. a numpy vector of length n, initialized a all-zeros.
                
        '''

        self.n = n
        self.e = e
        self.Q = np.zeros(self.n, dtype=np.float)
        self.c = np.zeros(self.n, dtype=np.int)


   # ----------------------------------------------
    def forward(self):
        '''
            The policy function of the agent.
            The agent will 
                (1) with a small probability (epsilon or e), randomly pull a lever with uniform distribution on all levers (Exploration); 
                (2) with a big probability (1-e) to pull the arm with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.
            Output:
                a: the index of the lever to pull. a is an integer scalar between 0 and n-1. 
        '''

        rand = np.random.random()
        if (rand < self.e):
            a = np.random.randint(self.n)  
        
        else:
            a = np.argmax(self.Q)

        return a


    #-----------------------------------------------------------------
    def update(self, a,r):
        '''
            Updating the parameters of the agent.
            (1) increase the count of lever
            (2) update the expected reward based upon the received reward r.
            Input:
                a: the index of the arm being pulled. a is an integer scalar between 0 and n-1. 
                r: the reward returned, a float scalar. 
        '''

        self.c[a] += 1
        self.Q[a] += (1./self.c[a]) * (r - self.Q[a])



    #-----------------------------------------------------------------
    def play(self, g, n_steps=1000):
        '''
            Play the game for n_steps steps. In each step,
            (1) pull a lever and receive the reward from the game
            (2) update the parameters 
            Input:
                g: the game machine, a multi-armed bandit object. 
                n_steps: number of steps to play in the game, an integer scalar. 
        '''

        
        for i in range(n_steps):
            a = self.forward()
            r = g.step(a)
            self.update(a,r)
