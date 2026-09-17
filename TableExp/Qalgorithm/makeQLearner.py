from Qalgorithm.DoubleQLearner import DoubleQLearner
from Qalgorithm.SingleQLearner import SingleQLearner
from Qalgorithm.DataMixDoubleQLearner import DataMixDoubleQLearner
from Qalgorithm.AveragedQLearner import AveragedQLearner
from Qalgorithm.MaxminQLearner import MaxminQLearner
from Qalgorithm.WeightedQLearner import WeightedQLearner
from Qalgorithm.EnsembleBootstrappedQLearner import EnsembleBootstrappedQLearner
from Qalgorithm.AdaptiveOrderQLearner import AdaptiveOrderQLearner
from Qalgorithm.CAQLearner import CAQLearner
from Qalgorithm.ErrorClipQLearner import ErrorClipQLearner
from Qalgorithm.ActionCandidateClippedQLearner import ActionCandidateClippedQLearner

def makeQLearner(cfg):
    if cfg.algorithm == "DoubleQ":
        agent = DoubleQLearner(gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon, learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "SingleQ":
        agent = SingleQLearner(gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon, learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "mixDQ":
        agent = DataMixDoubleQLearner(Mixratio=cfg.mixratio, gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon, learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "AveragedQ":
        agent = AveragedQLearner(K=cfg.K, gamma=cfg.gamma, epsilon=cfg.epsilon, adplearningRate=cfg.adplr, learningRate=cfg.lr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "MaxminQ":
        agent = MaxminQLearner(K=cfg.K, gamma=cfg.gamma, epsilon=cfg.epsilon, adplearningRate=cfg.adplr, learningRate=cfg.lr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "WeightedQ":
        agent = WeightedQLearner(c=cfg.c, gamma=cfg.gamma, epsilon=cfg.epsilon, adplearningRate=cfg.adplr, learningRate=cfg.lr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "EBQL":
        agent = EnsembleBootstrappedQLearner(K=cfg.K, gamma=cfg.gamma, epsilon=cfg.epsilon, adplearningRate=cfg.adplr, learningRate=cfg.lr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "AdaOQ":
        agent = AdaptiveOrderQLearner(M=cfg.M, m=cfg.m, gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon, learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "CAQ":
        agent = CAQLearner(gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon,
                                      learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp,
                                      epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd,
                                      Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "ECQ":
        agent = ErrorClipQLearner(delta=cfg.delta, gamma=cfg.gamma, epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon,
                                      learningRate=cfg.lr, adplearningRate=cfg.adplr, lrexp=cfg.lrexp,
                                      epsilonexp=cfg.epsilonexp, env=cfg.envobj, Qmean=cfg.Qmean, Qstd=cfg.Qstd,
                                      Qinitmode=cfg.Qinitmode)
    elif cfg.algorithm == "ACCDQ":
        agent = ActionCandidateClippedQLearner(multiplexratio=cfg.multiplexratio, topK=cfg.K, gamma=cfg.gamma,
                                               epsilon=cfg.epsilon, adpepsilon=cfg.adpepsilon, learningRate=cfg.lr,
                                               adplearningRate=cfg.adplr,
                                               lrexp=cfg.lrexp, epsilonexp=cfg.epsilonexp, env=cfg.envobj,
                                               Qmean=cfg.Qmean,
                                               Qstd=cfg.Qstd, Qinitmode=cfg.Qinitmode)
    else:
        raise NotImplementedError

    return agent