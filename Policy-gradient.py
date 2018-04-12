import math
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.optim import Adam 
import gym
from problem4 import Game, QNet
from torch.distributions import Categorical

#-------------------------------------------------------------------------
'''
    Problem 5: Policy-gradient Method for Deep Reinforcement Learning 
    In this problem, we will implement an AI player for the frozen lake game, using a neural network.
    Instead of using the neural network to approximate the Q function, we use the neural network to directly output the action. 
    The input (game state) is represented as the one-hot encoding. 
    The neural network has one fully connected layer (without biases) with softmax activation. 
    The outputs of the network are the probabilities of taking each action for the input state. 
    We will use backpropagation to train the neural network.
    ------------------------------
    Action code 
        0 : "LEFT",
        1 : "DOWN",
        2 : "RIGHT",
        3 : "UP"
'''

#-------------------------------------------------------
class PolicyNet(QNet):
    ''' 
        The agent is trying to maximize the sum of rewards (payoff) in the game using Policy-Gradient Method. 
        PolicyNet is a subclass of the agent in problem 4.
        We will use the weight matrix W as the policy network.
        The agent will use the output probabilities (after softmax) to randomly sample actions. 
    '''
    # ----------------------------------------------
    def __init__(self, n=4, d=16):
        ''' Initialize the agent. 
            Inputs:
                n: the number of actions in the game, an integer scalar. 
                d: the number of dimensions of the states of the game, an integer scalar. 
        '''
        super(PolicyNet, self).__init__(n,d,0.)

    # ----------------------------------------------
    def compute_z(self, s):
        '''
          Given a state of the game, computing the linear logits of neural netowrk for all actions. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length d. 
          Output:
                z: the linear logits of the network, a pytorch Variable of length n.  n is the number of actions in the game.
        '''

        z = th.matmul(self.W,s)

        return z


    #-----------------------------------------------------------------
    @staticmethod
    def compute_a(z):
        '''
            Computing probabilities of the agent taking each action. 
            Input:
                z: the linear logit of the neural network, a float variable of length n.
                    Here n is the number of actions. 
            Output:
                a: the probability of the agent taking different actions, a float variable of length n. 
        '''

        a = th.nn.functional.softmax(z,0)

        return a

    # ----------------------------------------------
    def forward(self, s):
        '''
          The policy function of the agent. 
          Inputs:
                s: the current state of the machine, a pytorch vector of length n_s. 
          Output:
                a: the probability of the agent taking different actions, a float variable of length n. 
        '''

        if np.random.rand() < self.e:
            return np.random.randint(self.n)
        
        z = self.compute_z(s)
      
        a = self.compute_a(z)

        return a


    #--------------------------
    @staticmethod
    def sample_action(a):
        '''
            Given a vector of activations (probabilities of taking each action), randomly sample an action according to the probabilities. 
            Input:
                a: the probabilities of different actions, a pytorch variable of length n.
            Output:
                m: the sampled action (move), an integer scalar of value, 0, 1, ..., n-1 
                logp: the log probability of the sampled action, a float Variable of value between 0 and 1
        '''

        action = Categorical(a)
        move = action.sample()
        logp = action.log_prob(move)
        m = move.data[0]       

        return m, logp 


    #--------------------------
    def play_episode(self, env, render =False):
        '''
            Play one episode of the game and collect data of actions and rewards, while fixing the model parameters.
            This process is also called roll-out or trial.
         
            At each step, sample an action randomly from the output of the network and collect the reward and new state from the game.
            An episode is over when the returned value for "done"= True.
            Input:
                env: the envirement of the game 
                render: whether or not to render the game (it's slower to render the game)
            Output:
                S: the game states, a python list of game states in each step. S[i] is the game state at the i-th step.
                M: the sampled actions in the game, a python list of sampled actions.
                    M[i] is the sampled action at the i-th step.
                logP: the log probability of sampled action at each step, a python list.
                    logP[i] is the log probability Variable in the i-th step.
                R: the raw rewards in the game, a python list of collected rewards.
                    R[i] is the collected reward at the i-th step.
        '''
        S,M,logP,R = [],[],[],[]
        s = env.reset() # initial state of the game 
        done = False
        # play the game until the episode is done
        while not done:
            if render:
                env.render() # render the game

            # compute the probability of taking each action
            z = self.compute_z(s)
            a = self.compute_a(z)
            # sample an action based upon the probabilities
            m, logp = self.sample_action(a)
            # play one step in the game
            S.append(s)
            M.append(m)
            logP.append(logp)
            new_s, r, done, _ = env.step(m)
            s = new_s
            R.append(r)

        return S,M,logP,R


    #--------------------------
    @staticmethod
    def discount_rewards(R,gamma=0.98):
        '''
            Given a time sequence of raw rewards in a game episode, computing discounted rewards (non-sparse) 
            Input:
                R: the raw rewards collected at each step of the game, a float python list of length h.
                gamma: discount factor, a float scalar between 0 and 1.
            Output:
                dR: discounted future rewards for each step, a float numpy array of length h.
        '''

        dR = []
        for i in range(len(R)):
            count = 1
            val = R[i]
            for j in range(i + 1, len(R)):
                val += (gamma ** count) * R[j]
                count += 1
            dR.append(val)

        return dR 
 
    #-----------------------------------------------------------------
    @staticmethod
    def compute_L(logP,dR):
        '''
            Computing policy loss of a game episode: the sum of (- log_probability * discounted_reward) at each step
            Input:
                logP: the log probability of sampled action at each step, a python list of length n.
                    Here n is the number of steps in the game.
                    logP[i] is the log probability Variable of the i-th step in the game.
                dR: discounted future rewards for each step, a float python list of length n.
            Output:
                L: the squared error of step, a float scalar Variable. 
        '''

        L = 0

        for i in range(len(dR)):
            L += -logP[i]*dR[i]

        return L 

 
    #--------------------------
    def play(self, env, n_episodes, render =False,gamma=.95, lr=.1):
        '''
            Given a game environment of gym package, play multiple episodes of the game.
            An episode is over when the returned value for "done"= True.
            At each step, pick an action and collect the reward and new state from the game.
            After an episode is done, compute the discounted reward and update the parameters of the model using gradient descent.
            Input:
                env: the envirement of the game of openai gym package 
                n_episodes: the number of episodes to play in the game. 
                render: whether or not to render the game (it's slower to render the game)
                gamma: the discount factor, a float scalar between 0 and 1.
                lr: learning rate, a float scalar, between 0 and 1.
            Outputs:
                total_rewards: the total number of rewards achieved in the game 
        '''
        optimizer = Adam([self.W], lr=lr)
        total_rewards = 0.
        # play multiple episodes
        for _ in xrange(n_episodes):

            _, _, logP, R = self.play_episode(env)

            dR = self.discount_rewards(R, gamma)

            L = self.compute_L(logP, dR)

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_rewards += sum(R) # assuming the list of rewards of the episode is R
        return total_rewards

