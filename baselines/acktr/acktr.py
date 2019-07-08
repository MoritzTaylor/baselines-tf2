import os.path as osp
import time
import functools
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.a2c.runner import Runner
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.acktr import kfac
from baselines.ppo2.ppo2 import safemean
from collections import deque


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', is_async=True):
        nbatch = nenvs * nsteps

        self.model = step_model = policy(nenvs, 1)
        self.model2 = train_model = policy(nbatch, nsteps)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, PG_LR: cur_lr, VF_LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy = calc_loss(obs,
                                                             A=actions,
                                                             ADV=advs,
                                                             R=rewards,
                                                             PG_LR=cur_lr,
                                                             VF_LR=cur_lr)

            return policy_loss, value_loss, policy_entropy

        self.train = train

        def calc_loss(obs, A, ADV, R, PG_LR, VF_LR):
            neglogpac = train_model.pd.neglogp(A)
            self.logits = train_model.pi

            pg_loss = tf.reduce_mean(ADV * neglogpac)
            entropy = tf.reduce_mean(train_model.pd.entropy())
            pg_loss = pg_loss - ent_coef * entropy
            vf_loss = tf.losses.mean_squared_error(tf.squeeze(train_model.vf), R)
            train_loss = pg_loss + vf_coef * vf_loss

            ##Fisher loss construction
            self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
            sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
            self.vf_fisher = vf_fisher_loss = - vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
            self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

            self.params = params = find_trainable_variables("acktr_model")

            self.grads_check = grads = tf.gradients(train_loss, params)

            with tf.device('/gpu:0'):
                self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip, \
                                                        momentum=0.9, kfac_update=1, epsilon=0.01, \
                                                        stats_decay=0.99, is_async=is_async, cold_iter=10,
                                                        max_grad_norm=max_grad_norm)

                # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
                optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
                train_op, q_runner = optim.apply_gradients(list(zip(grads, params)))

            return pg_loss, vf_loss, entropy,


def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True, **network_kwargs):
    set_global_seeds(seed)
    ...
