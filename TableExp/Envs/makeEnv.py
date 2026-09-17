from Envs.huberEnv import huberEnv
from Envs.multiarmEnv import multiarmEnv

def makeEnv(cfg):
    if cfg.env == "multiarmenv":
        env = multiarmEnv(arm_count=cfg.Narm, std=cfg.armstd)
    elif cfg.env == "huberenv":
        env = huberEnv(RewardNoiseProb=cfg.RewardNoiseprob, Narm=cfg.Narm, armstd=cfg.armstd)
    else:
        raise ValueError("Unknown env %s" % cfg.env)
    return env