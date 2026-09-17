import numpy as np
import math
from Qalgorithm.algoshareclass import QinitmodeEnum

class AveragedQLearner():
    def __init__(self, epsilon=1.0, gamma=0.95, Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,
                 adplearningRate=True, learningRate=1.0, K=4, lrexp=0.8, epsilonexp=0.5, env=None):
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.adplearningRate=adplearningRate
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon

        self.gamma = gamma
        self.K = K

        self.env = env
        self.Qinitmode = Qinitmode
        self.init_Q_tables(Qmean, Qstd)
        self.oldest_Q_table_index_S_A = np.zeros((self.env.nState, self.env.nAction), dtype=int)


    def init_Q_tables(self,Qmean=0, Qstd=0.01,):
        #np.random.seed(42)

        if self.Qinitmode == QinitmodeEnum.ALLSAME:
            self.Q_tables = np.random.normal(np.random.normal(Qmean, Qstd), 0,
                                             size=(self.K, self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.TABSAME:
            self.Q_tables = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.K, self.env.nState, self.env.nAction))
            for i in range(1, self.K):
                self.Q_tables[i] = np.random.normal(np.random.normal(Qmean, Qstd), 0,
                                                    size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.ALLDIFF:
            self.Q_tables = np.random.normal(Qmean, Qstd, size=(self.K, self.env.nState, self.env.nAction))
        else:
            raise NotImplementedError

        self.Count_S_A = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1

        epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:

            selector_indices = np.arange(self.K)
            selector_Q_tables = self.Q_tables[selector_indices, state, :action_number]
            selector_Q_bar = np.mean(selector_Q_tables, axis=0)
            action = np.argmax(selector_Q_bar)
            #action = np.argmax(self.Q_tables[0][state, :action_number])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):

        td_target = reward

        selector_indices, estimator_indices = np.arange(self.K), np.arange(self.K)

        if not done:
            # select optimal action for s_t+1
            action_number = self.env.action_number(next_state)
            Qavg = self.Q_tables[:, next_state, :action_number]
            Qavg = np.mean(Qavg, axis=0)


            estimate_max_Qsa = np.max(Qavg)

            td_target = reward + self.gamma * estimate_max_Qsa


        self.Count_S_A[state, action] += 1
        if self.adplearningRate:
            lr = self.learningRate / np.power(self.Count_S_A[state, action], self.lrexp)
        else:
            lr = self.learningRate
        # only 2 tables condition
        # self.Q_tables[0][state, action] = self.Q_tables[1][state, action].copy()
        # self.Q_tables[1][state, action] += lr * (td_target - self.Q_tables[1][state, action])

        oldest_Q_table_index = self.oldest_Q_table_index_S_A[state, action]
        newest_Q_table_index = oldest_Q_table_index -1 if oldest_Q_table_index > 0 else self.K - 1
        newQsa = self.Q_tables[newest_Q_table_index, state, action] + lr * (td_target - self.Q_tables[newest_Q_table_index, state, action])
        self.Q_tables[oldest_Q_table_index, state,action] = newQsa

        oldest_Q_table_index = (oldest_Q_table_index+1) % self.K
        self.oldest_Q_table_index_S_A[state, action] = oldest_Q_table_index




    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q_s_bar = np.mean(self.Q_tables[:, state, :action_number], axis=0)
        return max(Q_s_bar)

    def getQ_tables(self):
        return self.Q_tables

