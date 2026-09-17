import numpy as np
from Qalgorithm.algoshareclass import QinitmodeEnum


class CAQLearner:
    """Confidence-aware Q-learning with a per-(state, action) TD-error EMA.

    For the currently visited (s, a), let B_t(s, a) be the EMA of previous
    absolute TD errors.  The confidence is

        c_t = 1 / (1 + exp(K * (|delta_t| - lambda * B_t) / (B_t + eps)))

    B_t is used before it is updated with delta_t, so the current sample does
    not influence its own confidence decision.
    """

    def __init__(self, env=None, epsilon=1, adpepsilon=True, gamma=0.95, learningRate=1, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, tolerance=3.0,
                 sharpness=5.0, ema_decay=0.99, ema_epsilon=1e-6,
                 initial_td_baseline=1.0, Qmean=0, Qstd=0.01,Qinitmode=QinitmodeEnum.ALLSAME,):
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.adpepsilon = adpepsilon
        self.adplearningRate = adplearningRate
        self.tolerance = tolerance
        self.sharpness = sharpness
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.initial_td_baseline = initial_td_baseline
        self.env = env
        self.Qinitmode = Qinitmode
        self.init_Q_table(Qmean, Qstd)
        self.TD_EMA = np.full(
            (self.env.nState, self.env.nAction),
            self.initial_td_baseline,
            dtype=float,
        )

    def init_Q_table(self, Qmean=0, Qstd=0.01):


        if self.Qinitmode == QinitmodeEnum.ALLSAME:
            self.Q = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.TABSAME:
            self.Q = np.random.normal(np.random.normal(Qmean, Qstd), 0, size=(self.env.nState, self.env.nAction))
        elif self.Qinitmode == QinitmodeEnum.ALLDIFF:
            self.Q = np.random.normal(Qmean, Qstd, size=(self.env.nState, self.env.nAction))
        else:
            raise NotImplementedError


        self.Count_S_A = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1
        if self.adpepsilon:
            epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)
        else:
            epsilon_temp = self.epsilon

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:
            Q3 = [self.Q[state][i] for i in range(action_number)]
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(action_number)
        return action

    def confidence(self, td_error, td_baseline):
        relative_excess = (
            abs(td_error) - self.tolerance * td_baseline
        ) / (td_baseline + self.ema_epsilon)
        exponent = np.clip(self.sharpness * relative_excess, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(exponent))

    def update_td_ema(self, state, action, td_error):
        previous = self.TD_EMA[state, action]
        self.TD_EMA[state, action] = (
            self.ema_decay * previous
            + (1.0 - self.ema_decay) * abs(td_error)
        )

    def learning(self, state, action, reward, next_state, done):
        self.Count_S_A[state][action] += 1
        target = reward
        if not done:
            target += self.gamma * max(self.Q[next_state, :])

        td_error = target - self.Q[state, action]
        td_baseline = self.TD_EMA[state, action]
        confidence = self.confidence(td_error, td_baseline)

        lr = self.learningRate / np.power(self.Count_S_A[state][action], self.lrexp)
        #lr = 1
        self.Q[state, action] += lr * confidence * td_error
        # Only the visited (state, action) receives this historical update.
        self.update_td_ema(state, action, td_error)

    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q3 = [self.Q[state][i] for i in range(action_number)]

        return max(Q3)

    def getQ_tables(self):
        Q = self.Q.copy()[np.newaxis, :, :]
        return Q
