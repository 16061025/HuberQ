import numpy as np
#import matplotlib.pyplot as plt
from Qalgorithm.algoshareclass import QinitmodeEnum

class DataMixDoubleQLearner:
    def __init__(self, Mixratio=0.2, epsilon=1.0,
                 Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,
                 adpepsilon=True,  gamma=0.95, learningRate=1.0, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, env=None):
        self.UpdateABProbability = Mixratio
        self.UpdateAProbability = (1-self.UpdateABProbability)/2
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.adpepsilon = adpepsilon
        self.adplearningRate = adplearningRate
        self.env = env
        self.Qinitmode = Qinitmode
        self.init_Q_table(Qmean, Qstd)
        self.prob = None


    def init_Q_table(self,Qmean=0, Qstd=0.01,):

        if self.Qinitmode == QinitmodeEnum.ALLSAME:
            self.Q1 = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
            self.Q2 = self.Q1.copy()
        elif self.Qinitmode == QinitmodeEnum.TABSAME:
            self.Q1 = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
            self.Q2 = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.ALLDIFF:
            self.Q1 = np.random.normal(Qmean, Qstd, size=(self.env.nState, self.env.nAction))
            self.Q2 = np.random.normal(Qmean, Qstd, size=(self.env.nState, self.env.nAction))
        else:
            raise NotImplementedError

        self.Count_S_A_1 = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S_A_2 = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1
        #self.prob = np.random.random()
        if self.adpepsilon:
            epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)
        else:
            epsilon_temp = self.epsilon
        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:
            # if self.prob is None:
            #     Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            #     action = np.argmax(Q3[:action_number])
            # else:
            #     if self.prob < self.UpdateAProbability:
            #         Q3 = self.Q1[state]
            #     elif self.prob >= 1 - self.UpdateABProbability:
            #         Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            #     else:
            #         Q3 = self.Q2[state]
            #     action = np.argmax(Q3[:action_number])
            Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            action = np.argmax(Q3[:action_number])
        else:
            action = np.random.choice(action_number)
        return action


    def learning(self, state, action, reward, next_state, done):
        Y1 = reward
        Y2 = reward
        Y =reward
        self.prob = np.random.random()
        if self.prob < self.UpdateAProbability:
            self.Count_S_A_1[state][action] += 1
            if self.adplearningRate:
                lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], self.lrexp)
            else:
                lr_1 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y += self.gamma * self.Q2[next_state][np.argmax(self.Q1[next_state][:action_number])]
            self.Q1[state][action] += lr_1 * (Y - self.Q1[state][action])
        elif self.prob >= 1-self.UpdateABProbability:
            self.Count_S_A_1[state][action] += 1
            self.Count_S_A_2[state][action] += 1
            if self.adplearningRate:
                lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], self.lrexp)
                lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], self.lrexp)
            else:
                lr_1 = self.learningRate
                lr_2 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y1 += self.gamma * self.Q1[next_state][np.argmax(self.Q1[next_state][:action_number])]
                Y2 += self.gamma * self.Q2[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q1[state][action] += lr_1 * (Y1 - self.Q1[state][action])
            self.Q2[state][action] += lr_2 * (Y2 - self.Q2[state][action])
        else:
            self.Count_S_A_2[state][action] += 1
            if self.adplearningRate:
                lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], self.lrexp)
            else:
                lr_2 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y += self.gamma * self.Q1[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q2[state][action] += lr_2 * (Y - self.Q2[state][action])


    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
        return max(Q3)


    def getQ_tables(self):
        Q = np.stack((self.Q1, self.Q2), axis=0)
        return Q


