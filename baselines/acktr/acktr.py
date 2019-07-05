import os.path as osp
import time
import functools
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.a2c.runner import Runner
# from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.acktr import kfac
from baselines.ppo2.ppo2 import safemean
from collections import deque


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', is_async=True):
        nbatch = nenvs * nsteps

        self.model = step_model = policy(nenvs, 1)
        self.model2 = train_model = policy(nenvs * nsteps, nsteps)
        ...

    def train(obs, states, rewards, masks, actions, values):
        ...


def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True, **network_kwargs):
    set_global_seeds(seed)
    ...
