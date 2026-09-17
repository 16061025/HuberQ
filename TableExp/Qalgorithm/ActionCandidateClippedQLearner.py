import math

import numpy as np
from Qalgorithm.algoshareclass import QinitmodeEnum

class ActionCandidateClippedQLearner:
    def __init__(self, epsilon=1.0, adpepsilon=True,  gamma=0.95,
                 Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,
                 learningRate=1.0, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, multiplexratio=1, topK=1,env=None):
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.adpepsilon = adpepsilon
        self.adplearningRate = adplearningRate
        self.env = env
        self.multiplexratio = multiplexratio
        self.topK = topK
        self.Qinitmode = Qinitmode
        self.init_Q_table(Qmean, Qstd)


    def init_Q_table(self, Qmean=0, Qstd=0.01):

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
        if self.adpepsilon:
            epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)
        else:
            epsilon_temp = self.epsilon
        action_number = self.env.action_number(state)

        if np.random.random() >= epsilon_temp:

            Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            action = np.argmax(Q3)
            # if self.prob >= 0.5:
            #     action = np.argmax(self.Q1[state, :action_number])
            # else:
            #     action = np.argmax(self.Q2[state, :action_number])
        else:
            action = np.random.choice(action_number)
        return action


    def learning(self, state, action, reward, next_state, done):

        Y = reward
        self.prob = np.random.random()
        if self.prob >= 0.5:
            self.Count_S_A_1[state][action] += 1
            if self.adplearningRate:
                lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], self.lrexp)
            else:
                lr_1 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                #self.topK = int(round(self.multiplexratio*(action_number-1)+1, 0))
                optimalActionSet = np.argpartition(-1* self.Q2[next_state][:action_number], self.topK-1)[:self.topK]

                optimalActionindexinSet = np.argmax(self.Q1[next_state][optimalActionSet])
                optimalAction = optimalActionSet[optimalActionindexinSet]
                Q2estimation=self.Q2[next_state][optimalAction]
                Q1estimation = np.max(self.Q1[next_state][:action_number])


                Y += self.gamma *  min(Q1estimation, Q2estimation)
            self.Q1[state][action] += lr_1 * (Y - self.Q1[state][action])
        else:
            self.Count_S_A_2[state][action] += 1
            if self.adplearningRate:
                lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], self.lrexp)
            else:
                lr_2 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                #self.topK = int(round(self.multiplexratio*(action_number-1)+1, 0))
                optimalActionSet = np.argpartition(-1 * self.Q1[next_state][:action_number], self.topK - 1)[:self.topK]

                optimalActionindexinSet = np.argmax(self.Q2[next_state][optimalActionSet])
                optimalAction = optimalActionSet[optimalActionindexinSet]
                Q1estimation = self.Q1[next_state][optimalAction]
                Q2estimation = np.max(self.Q2[next_state][:action_number])

                Y += self.gamma *  min(Q1estimation, Q2estimation)
            self.Q2[state][action] += lr_2 * (Y - self.Q2[state][action])

    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
        return max(Q3)

    def getQ_tables(self):
        Q = np.stack((self.Q1, self.Q2), axis=0)
        return Q
