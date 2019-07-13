import os.path as osp
import time
import functools
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.policies import PolicyWithValue

from baselines.a2c.runner import Runner
from baselines.a2c.utils import LinearTimeDecay #Scheduler
from baselines.acktr import kfac
from baselines.ppo2.ppo2 import safemean
from collections import deque


class Model(tf.keras.Model):

    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', is_async=True):
        super(Model, self).__init__(name='ACKTRModel')

        nbatch = nenvs * nsteps

        # TODO
        #self.model = step_model = policy(nenvs, 1)
        #self.model2 = train_model = policy(nbatch, nsteps)
        train_model = PolicyWithValue(ac_space, policy, value_network=None, estimate_q=False)

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        self.kfac_clip = kfac_clip
        self.lr = lr
        #self.lrschedule = lrschedule
        self.lrschedule = LinearTimeDecay(lr)
        self.is_asybc = is_async
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps

        #self.save = functools.partial(save_variables)
        #self.load = functools.partial(load_variables)
        self.train_model = train_model
        #self.step_model = step_model
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state

    @tf.function
    def train(self, obs, states, rewards, masks, actions, values):
        advs = rewards - values

        with tf.GradientTape() as tape:
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            neglogpac = self.train_model.pd.neglogp(actions)
            self.logits = self.train_model.pi

            # TODO: states and masks like in the original implementation?
            pg_loss = tf.reduce_mean(advs * neglogpac)
            entropy = tf.reduce_mean(self.train_model.pd.entropy())
            pg_loss = pg_loss - self.ent_coef * entropy
            vf_loss = tf.losses.mean_squared_error(tf.squeeze(self.train_model.vf), rewards)

        train_loss = pg_loss + self.vf_coef*vf_loss

        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
        sample_net = self.train_model.vf + tf.random_normal(tf.shape(self.train_model.vf))
        self.vf_fisher = vf_fisher_loss = - self.vf_fisher_coef * tf.reduce_mean(
            tf.pow(self.train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params = params = tape.watched_variables() #before: find_trainable_variables("acktr_model")

        self.grads_check = grads = tf.gradient(train_loss, params)

        with tf.device('/gpu:0'):
            # TODO: Everytime a new optim? Maybe adding 'learning_rate' param somewhere else?
            self.optim = optim = kfac.KfacOptimizer(learning_rate=cur_lr, clip_kl=self.kfac_clip, \
                                                    momentum=0.9, kfac_update=1, epsilon=0.01, \
                                                    stats_decay=0.99, is_async=self.is_async, cold_iter=10,
                                                    max_grad_norm=self.max_grad_norm)

            # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads, params)))

        self.q_runner = q_runner
        #self.lr = self.lrschedule()
        #self.lr = Scheduler(v=self.lr, nvalues=self.total_timesteps, schedule=self.lrschedule)

        return pg_loss, vf_loss, entropy


def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True, **network_kwargs):
    set_global_seeds(seed)
    ...
