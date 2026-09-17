import argparse
from Qalgorithm.algoshareclass import QinitmodeEnum

def str_to_bool(value):

    if value in ["True"]:
        return True
    elif value in ["False"]:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def str_to_QinitEnum(value):
    if value in ["ALLSAME"]:
        return QinitmodeEnum.ALLSAME
    elif value in ["ALLDIFF"]:
        return QinitmodeEnum.ALLDIFF
    elif value in ["TABSAME"]:
        return QinitmodeEnum.TABSAME
    else:
        raise argparse.ArgumentTypeError(f'Invalid Qinitmode: {value}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--K', default=6, type=int)
    parser.add_argument('--M', default=4, type=int)
    parser.add_argument('--m', default=2, type=int) #for OrderQ
    parser.add_argument('--c', default=10, type=float) # for WeightedQ
    parser.add_argument('--adpc', type=str_to_bool, default=True)  # for WeightedQ
    parser.add_argument('--N_selector', default=1, type=int)
    parser.add_argument('--N_estimator', default=1, type=int)
    parser.add_argument('--mixratio', default=0.5, type=float)
    parser.add_argument('--multiplexratio', default=0.5, type=float)
    parser.add_argument('--algorithm', default="EBQ", type=str)
    parser.add_argument('--env', default="env", type=str)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--adplr', type=str_to_bool, default=True)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--adpepsilon', type=str_to_bool, default=True)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--lrexp', default=0.8, type=float)
    parser.add_argument('--epsilonexp', default=0.5, type=float)
    parser.add_argument('--total_repeat', default=1000, type=int)
    parser.add_argument('--submean', default=-0.1, type=float)
    parser.add_argument('--optimalmean', default=0, type=float)
    parser.add_argument('--subvar', default=1, type=float)
    parser.add_argument('--optimalvar', default=1, type=float)

    parser.add_argument('--steps', default=10000, type=int)
    parser.add_argument('--episodes', default=10000, type=int)
    parser.add_argument('--max_episode_len', default=100, type=int)
    parser.add_argument('--eval_steps', default=20, type=int)
    parser.add_argument('--eval_episodes', default=100, type=int)
    parser.add_argument('--use_episode', type=str_to_bool, default=False)

    parser.add_argument('--R_selector', default=1.0, type=float)
    parser.add_argument('--R_overlap', default=1.0, type=float)

    parser.add_argument('--nrow', default=3, type=int)
    parser.add_argument('--ncol', default=3, type=int)
    parser.add_argument('--gridr1', default=-6, type=float)
    parser.add_argument('--gridr2', default=4, type=float)
    parser.add_argument('--gridgr1', default=-30, type=float)
    parser.add_argument('--gridgr2', default=40, type=float)

    parser.add_argument('--optimalaction', type=str_to_bool, default=False)
    parser.add_argument('--sel_update', type=str_to_bool, default=True)
    parser.add_argument('--info', default="no", type=str)

    parser.add_argument('--Narm', default=10, type=int)
    parser.add_argument('--armstd', default=10, type=float)

    parser.add_argument('--Qmean', default=20, type=float)
    parser.add_argument('--Qstd', default=1, type=float)
    parser.add_argument('--Qinitmode', type=str_to_QinitEnum, default=QinitmodeEnum.ALLSAME)
    # parser.add_argument('--device', default="cpu", type=str)
    # parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--RewardNoiseprob', default=0.01, type=float)

    parser.add_argument('--delta', default=0.5, type=float)

    args = parser.parse_args()




    return args