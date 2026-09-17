import numpy as np
import math

class multiarmEnv():
    '''
    for data multiplex
    '''

    def __init__(self, arm_count=10, std=10):
        self.arm_count = arm_count
        self.std = std
        self.state = 0


        self.nState = 1
        self.nAction = arm_count
        self.STATE_S = 0
        self.Right = 0
        one_sample = np.array([1.0])
        uniform_samples = np.random.rand(self.arm_count-1)
        self.mus = np.concatenate([one_sample, uniform_samples])


    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        mu = 1/(1+action)
        reward = np.random.normal(mu, self.std)
        return self.state, reward, False



    def action_number(self, state):
        return self.nAction
