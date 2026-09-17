import numpy as np

class huberEnv():
    def __init__(self, RewardNoiseProb=0.01, Narm=10, armstd=1):
        self.STATE_S = 0
        self.nState = 1
        self.nAction = Narm
        self.state = self.STATE_S
        self.reward_noise_prob = RewardNoiseProb
        self.armstd = armstd

    def reset(self):
        self.state = self.STATE_S
        return self.state

    def step(self, action):
        mu = 1/(action+1)
        if self.reward_noise_prob ==0:
            reward = np.random.normal(mu, 1)
            self.state = self.STATE_S
            return self.state, reward, False

        is_contaminated = np.random.random() < self.reward_noise_prob
        if is_contaminated:
            reward = np.random.normal(mu, 100)
        else:
            reward = np.random.normal(mu, self.armstd)
        self.state = self.STATE_S

        return self.state, reward, False
        # mu = 1 / (action + 1)
        #
        # reward = np.random.normal(mu, 10)
        # self.state = self.STATE_S
        #
        # return self.state, reward, False

    def action_number(self, state):
        return self.nAction
