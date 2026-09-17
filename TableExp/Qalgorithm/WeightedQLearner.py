import numpy as np
import math
from Qalgorithm.algoshareclass import QinitmodeEnum


class WeightedQLearner():
    def __init__(self, epsilon=1.0, gamma=0.95, Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,
                 adplearningRate=True, learningRate=1.0, lrexp=0.8, epsilonexp=0.5, c=10, adpc=True, env=None):
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.adplearningRate=adplearningRate
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.c = c
        self.adpc = adpc
        self.env = env
        self.Qinitmode = Qinitmode
        self.init_Q_tables(Qmean, Qstd)

    def init_Q_tables(self, Qmean=0, Qstd=0.01):
        # np.random.seed(42)

        if self.Qinitmode == QinitmodeEnum.ALLSAME:
            self.QU = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
            self.QV = self.QU.copy()
        elif self.Qinitmode == QinitmodeEnum.TABSAME:
            self.QU = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
            self.QV = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.ALLDIFF:
            self.QU = np.random.normal(Qmean, Qstd, size=(self.env.nState, self.env.nAction))
            self.QV = np.random.normal(Qmean, Qstd, size=(self.env.nState, self.env.nAction))
        else:
            raise NotImplementedError

        self.Count_S_A_U = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S_A_V = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S_A_single = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:
            #Q3 = [(self.QU[state][i] + self.QV[state][i]) / 2.0 for i in range(action_number)]
            Q3 = (self.QU[state][:action_number] + self.QV[state][:action_number]) / 2.0
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):

        td_target = reward

        if np.random.rand() >= 0.5:
            self.updateU, self.updateV = True, False
        else:
            self.updateU, self.updateV = False, True

        if not done:
            # select optimal action for s_t+1
            action_number = self.env.action_number(next_state)
            if self.updateU:
                if self.adpc:
                    optimal_a = np.argmax(self.QU[next_state][:action_number])
                    a_L = np.argmin(self.QU[next_state][:action_number])

                    Q_V_next_s_optimal_a = self.QV[next_state][optimal_a]
                    Q_V_next_s_a_L = self.QV[next_state][a_L]
                    abserr = np.abs(Q_V_next_s_optimal_a - Q_V_next_s_a_L)

                    Q_U_next_s_optimal_a = self.QU[next_state][optimal_a]
                    beta_U = abserr / (self.c + abserr)
                else:
                    beta_U = self.c


                td_target = reward + self.gamma * (beta_U * Q_U_next_s_optimal_a + (1-beta_U) * Q_V_next_s_optimal_a)


            elif self.updateV:
                if self.adpc:
                    optimal_a = np.argmax(self.QV[next_state][:action_number])
                    a_L = np.argmin(self.QV[next_state][:action_number])

                    Q_U_next_s_optimal_a = self.QU[next_state][optimal_a]
                    Q_U_next_s_a_L = self.QU[next_state][a_L]
                    abserr = np.abs(Q_U_next_s_optimal_a - Q_U_next_s_a_L)

                    Q_V_next_s_optimal_a = self.QV[next_state][optimal_a]
                    beta_V = abserr / (self.c + abserr)
                else:
                    beta_V = self.c

                td_target = reward + self.gamma * (beta_V * Q_V_next_s_optimal_a + (1 - beta_V) * Q_U_next_s_optimal_a)

        if self.updateU:
            self.Count_S_A_U[state][action] += 1
            if self.adplearningRate:
                lr = self.learningRate / np.power(self.Count_S_A_U[state][action], self.lrexp)
            else:
                lr = self.learningRate
            self.QU[state][action] += lr * (td_target - self.QU[state][action])
        elif self.updateV:
            self.Count_S_A_V[state][action] += 1
            if self.adplearningRate:
                lr = self.learningRate / np.power(self.Count_S_A_V[state][action], self.lrexp)
            else:
                lr = self.learningRate
            self.QV[state][action] += lr * (td_target - self.QV[state][action])


    def maxQ(self, state):
        action_number = self.env.action_number(state)
        #Q3 = [(self.QU[state][i] + self.QV[state][i]) / 2.0 for i in range(action_number)]
        Q3 = (self.QU[state][:action_number] + self.QV[state][:action_number]) / 2.0
        return max(Q3)

    def getQ_tables(self):
        Q = np.stack((self.QU, self.QV), axis=0)
        return Q








