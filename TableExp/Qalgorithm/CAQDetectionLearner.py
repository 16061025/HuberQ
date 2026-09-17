import numpy as np


class CAQDetectionLearner:
    """CAQ learner that additionally exposes its soft anomaly score for evaluation.

    The returned anomaly_score is 1 - confidence.  It is only recorded by the
    experiment runner; the learning update never receives a contamination label.
    """

    def __init__(self, env, epsilon, gamma, learningRate, m=0.1, n=0.1):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.m = m
        self.n = n
        self.env = env
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(self.env.nState, self.env.nAction))
        self.Q_gap = np.zeros((self.env.nState, self.env.nAction))

    def explore(self, state):
        if np.random.random() >= self.epsilon:
            return np.argmax(self.Q[state, :])
        return np.random.choice(len(self.Q[state, :]))

    def confidence(self, td_error, q_gap):
        score = -self.m * abs(td_error) + self.n * abs(q_gap)
        score = np.clip(score, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-score))

    def learning(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * max(self.Q[next_state, :])

        td_error = target - self.Q[state, action]
        q_gap = self.Q_gap[state, action]
        confidence = self.confidence(td_error, q_gap)

        q_change = self.learningRate * confidence * td_error
        self.Q[state, action] += q_change
        self.Q_gap[state, action] = abs(q_change)

        return {
            "confidence": float(confidence),
            "anomaly_score": float(1.0 - confidence),
            "td_error": float(td_error),
            "q_gap": float(q_gap),
            "update_magnitude": float(abs(q_change)),
        }

    def maxQ(self, state):
        return max(self.Q[state, :])
