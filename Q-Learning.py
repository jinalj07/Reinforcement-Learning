import math
import numpy as np
from problem2 import Agent
import gym
#-------------------------------------------------------------------------
'''
    Problem 3: Reinforcement Learning Problem (Q-Learning) 
    In this problem, we will implement an AI player for the frozen lake game.
    The main goal of this problem is to get familiar with reinforcement learning problem, and how to train the model parameters on the data from a game.
    We will use Q-learning to solve the problem.
    The detailed information about the frozen lake game can be found here: https://gym.openai.com/envs/FrozenLake-v0/
    
    ------------------------------
    Action code 
        0 : "LEFT",
        1 : "DOWN",
        2 : "RIGHT",
        3 : "UP"
'''


#-------------------------------------------------------
class QLearner(Agent):
    '''The agent is trying to maximize the sum of rewards (payoff) in the game using Q-Learning method. 
       The agent will 
                (1) with a small probability (epsilon or e), randomly choose an action with a uniform distribution on all actions (Exploration); 
                (2) with a big probability (1-e) to choose the action with the largest expected reward (Exploitation). If there is a tie, pick the one with the smallest index.'''
    # ----------------------------------------------
    def __init__(self, n=4, n_s=16, e=0.1):
        ''' Initializing the agent. 
            Inputs:
                n: the number of actions in the game, an integer scalar. 
                n_s: the number of states of the game, an integer scalar. 
                e: (epsilon) the probability of the agent randomly choosing an action with uniform probability. e is a float scalar between 0. and 1. 
            Outputs:
                self.n: the number of actions, an integer scalar. 
                self.e: the probability of the agent randomly choosing an action with uniform probability. e is a float scalar between 0. and 1. 
                self.Q: the expected discounted rewards for choosing each action at each state, a numpy matrix of shape n_s by n. We initialize the matrix with all-zeros.
                 QLearner is a subclass of the agent in problem 2.
        '''

        self.n, self.n_s, self.Q, self.e = n, n_s, np.zeros((n_s, n), dtype=np.float), e


    #-----------------------------------------------------------------
    def update(self,s,a,r,s_new, gamma=.95, lr=.1):
        '''
            Estimating the parameters of the agent using Bellman Equation.
            Q(s,a) = Expectation of ( r + gamma * max(Q(s_new,a')) )
            Update the Q parameters using gradient descent on the Bellman Equation. 
            Input:
                s: the current state of the game, an integer scalar. 
                a: the index of the action being chosen. a is an integer scalar between 0 and n-1. 
                r: the reward returned by the game, a float scalar. 
                s_new: the next state of the game, an integer scalar. 
                gamma: the discount factor, a float scalar between 0 and 1.
                lr: learning rate, a float scalar, between 0 and 1.
        '''

        self.Q[s][a] = (1-lr)*self.Q[s][a] + lr*( r + gamma * max(self.Q[s_new][:]))



 
    #--------------------------
    def play(self, env, n_episodes, render =False,gamma=.95, lr=.1):
        '''
            Given a game environment of gym package, play multiple episodes of the game.
            An episode is over when the returned value for "done"= True.
            (1) at each step, pick an action and collect the reward and new state from the game.
            (2) update the parameters of the model
            Input:
                env: the environment of the frozen-lake game of openai gym package 
                n_episodes: the number of episodes to play in the game. 
                render: whether or not to render the game (it's slower to render the game)
                gamma: the discount factor, a float scalar between 0 and 1.
                lr: learning rate, a float scalar, between 0 and 1.
            Outputs:
                total_rewards: the total number of rewards achieved in the game 
        '''
        total_rewards = 0.
        # play multiple episodes
        for _ in xrange(n_episodes):
            s = env.reset() # initialize the episode 
            done = False
            # play the game until the episode is done
            while not done:
                if render:
                    env.render() # render the game

                # agent selects an action
                act_taken = self.forward(s)
                # game return a reward and new state
                s_new, r, done, _ = env.step(act_taken)
                # agent update the parameters
                self.update(s, act_taken, r, s_new, gamma, lr)
                s = s_new

                total_rewards += r # assuming the reward of the step is r
        return total_rewards
