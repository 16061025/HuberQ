import numpy as np
from Qalgorithm.algoshareclass import QinitmodeEnum

class AdaptiveOrderQLearner:
    def __init__(self, M=2, m=2, epsilon=1.0, adpepsilon=True,  gamma=0.95,
                 Qmean=0, Qstd=0.01, Qinitmode=QinitmodeEnum.ALLSAME,
                 learningRate=1.0, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, env=None):

        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.adpepsilon = adpepsilon
        self.adplearningRate = adplearningRate
        self.env = env
        self.M = M
        self.m = m
        self.Qinitmode = Qinitmode
        self.init_Q_tables(Qmean, Qstd)

    def init_Q_tables(self, Qmean=0, Qstd=0.01):
        #np.random.seed(42)


        if self.Qinitmode == QinitmodeEnum.ALLSAME:
            self.Q_tables = np.random.normal(np.random.normal(Qmean, Qstd), 0,
                                             size=(self.M, self.env.nState, self.env.nAction))
        elif self.Qinitmode ==QinitmodeEnum.TABSAME:
            self.Q_tables = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.M, self.env.nState, self.env.nAction))
            for i in range(1, self.M):
                self.Q_tables[i] = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.ALLDIFF:
            self.Q_tables = np.random.normal(Qmean, Qstd, size=(self.M, self.env.nState, self.env.nAction))
        else:
            raise NotImplementedError

        self.Count_S_A = np.zeros(shape=(self.M, self.env.nState, self.env.nAction))
        self.Count_S_A_single = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1

        epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:
            action =  np.partition(self.Q_tables[:, state, :action_number], self.m - 1, axis=0)[self.m - 1].argmax()
            #action = np.mean(self.Q_tables[:, state, :action_number], axis=0).argmax()
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):

        Y = reward
        update_table_index = np.random.choice(self.M)
        self.Count_S_A[update_table_index][state][action] += 1
        if self.adplearningRate:
            lr_1 = self.learningRate / np.power(self.Count_S_A[update_table_index][state][action], self.lrexp)
        else:
            lr_1 = self.learningRate
        if not done:
            action_number = self.env.action_number(next_state)

            reorderQ = np.partition(self.Q_tables[:, next_state, :action_number], self.m - 1, axis=0)
            Qproxy = reorderQ[self.m - 1]
            qsaNext0 = Qproxy.max()

            qsaNext = np.partition(self.Q_tables[:, next_state, :action_number], self.m - 1, axis=0)[self.m - 1].max()
            Y += self.gamma * qsaNext

        self.Q_tables[update_table_index][state][action] += lr_1 * (Y - self.Q_tables[update_table_index][state][action])


    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q_s_bar = np.mean(self.Q_tables[:, state, :action_number], axis=0)
        return max(Q_s_bar)

    def getQ_tables(self):
        return self.Q_tables

