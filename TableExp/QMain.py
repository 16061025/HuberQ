import os

import numpy as np

from Logger import Logger
import time
from Envs.makeEnv import makeEnv
from Qalgorithm.makeQLearner import makeQLearner
from tqdm import tqdm
from arguments import parse_args


def QLearning(cfg):
    total_repeat = cfg.total_repeat

    max_Q_S_repeat = np.zeros((total_repeat, cfg.steps//100))
    Prsright = np.zeros((total_repeat, cfg.steps//100))
    AvgReward = np.zeros((total_repeat, cfg.steps//100))

    for repeat in tqdm(range(total_repeat)):
        env = makeEnv(cfg)
        cfg.envobj = env
        agent = makeQLearner(cfg)
        if repeat %100 == 0:
            log.logger.info(agent.getQ_tables())

        metrics = QstepUpdate(env=env, agent=agent, cfg=cfg)
        max_Q_S_repeat[int(repeat)] = metrics['max_Q_S']
        Prsright[int(repeat)] = metrics['Prsright']
        AvgReward[int(repeat)] = metrics['AvgReward']


        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_S is: \n{}".format(list(max_Q_S_repeat.mean(axis=0))))
    log.logger.info("Prsright is: \n{}".format(list(Prsright.mean(axis=0))))
    log.logger.info("AvgReward is: \n{}".format(list(AvgReward.mean(axis=0))))


def QstepUpdate(env,agent,cfg):
    max_Q_S = np.zeros(cfg.steps // 100)
    Prsright = np.zeros(cfg.steps // 100)
    AvgReward = np.zeros(cfg.steps // 100)


    total_reward = 0
    visit_s = 0
    visit_sright = 0
    state = env.reset()
    done = False


    for step in range(cfg.steps):
        if done:
            state = env.reset()
            done = False

        action = agent.explore(state)

        next_state, reward, done = env.step(action)

        total_reward += reward
        if step % 100 == 0:
            max_Q_S[step // 100] = agent.maxQ(env.STATE_S)
            AvgReward[step // 100] = total_reward / (step+1)

        agent.learning(state, action, reward, next_state, done)

        state = next_state

    metrics = {"max_Q_S": max_Q_S,
               "Prsright": Prsright,
               "AvgReward": AvgReward,
               }

    return metrics



if __name__ == "__main__":
    #np.random.seed(2025)
    # for i in range(3000):
    #     a = np.random.choice([-30,40])
    #     print(a)

    cfg = parse_args()

    if cfg.env in ["rightenv", "ABCenv", "ABenv"]:
        logdir = os.path.join("Result", f'{cfg.env}lr{cfg.lrexp}gamma{cfg.gamma}sm{cfg.submean}sv{cfg.subvar}om{cfg.optimalmean}ov{cfg.optimalvar}adplr{cfg.adplr}', cfg.algorithm)
    elif cfg.env in ["grid"]:
        logdir = os.path.join("Result", f'{cfg.env}lr{cfg.lrexp}gamma{cfg.gamma}Nrow{cfg.nrow}r{cfg.gridr1}r{cfg.gridr2}gr{cfg.gridgr1}gr{cfg.gridgr2}adplr{cfg.adplr}', cfg.algorithm)
    elif cfg.env in ["cliffenv"]:
        logdir = os.path.join("Result", f'{cfg.env}lr{cfg.lrexp}gamma{cfg.gamma}Nrow{cfg.nrow}r{cfg.gridr1}gr{cfg.gridgr1}d{cfg.gridr2}adplr{cfg.adplr}',
                              cfg.algorithm)
    elif cfg.env in ["SameActionenv"]:
        logdir = os.path.join("Result", f'{cfg.env}', cfg.algorithm)
    elif cfg.env in ["multiarmenv"]:
        logdir = os.path.join("Result", f'{cfg.env}lr{cfg.lrexp}gamma{cfg.gamma}Na{cfg.Narm}as{cfg.armstd}Qm{cfg.Qmean}Qs{cfg.Qstd}',cfg.algorithm)
    elif cfg.env in ["huberenv"]:
        logdir = os.path.join("Result", f'{cfg.env}Prob{cfg.RewardNoiseprob}Na{cfg.Narm}as{cfg.armstd}Qm{cfg.Qmean}Qs{cfg.Qstd}', cfg.algorithm)
    else:
        raise NotImplemented

    if not os.path.exists(logdir):
        try:
            os.makedirs(logdir)
        except OSError as e:
            pass

    if cfg.algorithm in ["rhoFixOverQ"]:
        if cfg.sel_update:
            update_Q = "sel"
        else:
            update_Q = "est"
        logfilename = f'log.Qenv{cfg.env}K{cfg.K}M{cfg.M}{update_Q}_update {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["mixAvgDQ", "mixDQ"]:
        logfilename = f'log.Qenv{cfg.env}mr{cfg.mixratio} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["AMultiplexQ"]:
        logfilename = f'log.Qenv{cfg.env}mr{cfg.multiplexratio}K{cfg.K} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ['AveragedQ', 'EnsembleQ', "EBQL", "KQ", "MaxminQ"]:
        logfilename = f'log.Qenv{cfg.env}K{cfg.K} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["WeightedQ"]:
        logfilename = f'log.Qenv{cfg.env}c{cfg.c} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["AdaOQ"]:
        logfilename = f'log.Qenv{cfg.env}M{cfg.M}m{cfg.m} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["ECQ"]:
        logfilename = f'log.Qenv{cfg.env}delta{cfg.delta} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    elif cfg.algorithm in ["ACCDQ"]:
        logfilename = f'log.Qenv{cfg.env}mr{cfg.multiplexratio}K{cfg.K} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'
    else:
        logfilename = f'log.Qenv{cfg.env} {time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}'

    cfg.logdir = logdir
    cfg.logfilename = logfilename

    log = Logger(
        os.path.join(logdir, logfilename),
        level='debug')
    log.logger.info(cfg)
    QLearning(cfg)

