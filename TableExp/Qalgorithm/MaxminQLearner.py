import numpy as np
import math
from Qalgorithm.algoshareclass import QinitmodeEnum



class MaxminQLearner():
    def __init__(self, epsilon=1.0, gamma=0.95, Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,
                 adplearningRate=True, learningRate=1.0,  K=2, S=1, lrexp=0.8, epsilonexp=0.5, env=None):
        self.learningRate = learningRate
        self.adplearningRate = adplearningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon

        self.gamma = gamma
        self.K = K
        self.S = S


        self.env = env
        self.Qinitmode = Qinitmode
        self.init_Q_tables(Qmean, Qstd)


    def init_Q_tables(self, Qmean=0, Qstd=0.01):
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

        self.Count_S_A = np.zeros(shape=(self.K, self.env.nState, self.env.nAction))
        self.Count_S_A_single = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:

            # selector_indeices = self.get_selector_indices()
            estimator_indices = np.arange(self.K)
            estimator_Q_tables = self.Q_tables[estimator_indices, state, :action_number]
            Qmin = np.min(estimator_Q_tables, axis=0)
            action = np.argmax(Qmin)
        else:
            action = np.random.choice(action_number)
        return action



    def learning(self, state, action, reward, next_state, done):

        td_target = reward

        selector_indices = np.random.choice(np.arange(self.K), size=self.S, replace=False)
        estimator_indices = np.arange(self.K)

        if not done:
            # select optimal action for s_t+1
            action_number = self.env.action_number(next_state)
            estimator_Q_tables = self.Q_tables[estimator_indices, next_state, :action_number]
            Qmin = np.min(estimator_Q_tables, axis=0)

            estimate_max_Qsa = np.max(Qmin)


            td_target = reward + self.gamma * estimate_max_Qsa


        self.Count_S_A[selector_indices, state, action] += 1
        if self.adplearningRate:
            lr = self.learningRate / np.power(self.Count_S_A[selector_indices, state, action], self.lrexp)
        else:
            lr = self.learningRate

        self.Q_tables[selector_indices, state, action] += lr * (td_target - self.Q_tables[selector_indices, state, action])



    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q_s_bar = np.mean(self.Q_tables[:, state, :action_number], axis=0)
        return max(Q_s_bar)
        # estimator_indices = np.arange(self.K)
        # estimator_Q_tables = self.Q_tables[estimator_indices, state, :action_number]
        # Qmin = np.min(estimator_Q_tables, axis=0)
        # maxQ = np.max(Qmin)
        # return maxQ

    def getQ_tables(self):
        return self.Q_tables

