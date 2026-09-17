import numpy as np

class bihuberEnv():
    def __init__(self, nAction =10, muA=0, muB=-0.1, sigmaAB= 10, RewardNoiseProb=0.01):
        self.STATE_S = 0
        self.STATE_A = 1
        self.STATE_B = 2
        self.STATE_T = 3
        self.nState = 3
        self.nAction = [2, nAction,nAction]
        self.state = self.STATE_S
        self.reward_noise_prob = RewardNoiseProb
        self.muA = muA
        self.muB = muB
        self.sigmaAB = sigmaAB
        self.actionA = 0
        self.actionB = 1

    def reset(self):
        self.state = self.STATE_S
        return self.state

    def step(self, action):
        if self.state == self.STATE_S:
            is_contaminated = np.random.random() < self.reward_noise_prob
            if is_contaminated:
                reward = np.random.normal(0, 100)
            else:
                reward = np.random.normal(0, 1)
            if action == self.actionA:
                next_state = self.STATE_A
            else:
                next_state = self.STATE_B
            done = False

        elif self.state == self.STATE_A:
            reward = np.random.normal(self.muA, self.sigmaAB)
            next_state = self.STATE_S
            done = True

        elif self.state == self.STATE_B:
            reward = np.random.normal(self.muB, self.sigmaS)
            next_state = self.STATE_S
            done = True
        self.state = next_state

        return self.state, reward, done


    def action_number(self, state):
        return self.nAction[state]
