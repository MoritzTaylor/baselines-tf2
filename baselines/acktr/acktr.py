import os.path as osp
import time
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.models import get_network_builder
from baselines.common.policies import PolicyWithValue

from baselines.a2c.runner import Runner
from baselines.a2c.utils import LinearTimeDecay
from baselines.acktr import kfac
from baselines.ppo2.ppo2 import safemean
from collections import deque


class Model(tf.keras.Model):

    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs,
                 total_timesteps,
                 nprocs=32,
                 nsteps=20,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 vf_fisher_coef=1.0,
                 lr=0.25,
                 max_grad_norm=0.5,
                 kfac_clip=0.001,
                 lrschedule='linear',
                 is_async=True):
        super(Model, self).__init__(name='ACKTRModel')

        nbatch = nenvs * nsteps

        # TODO: PolicyWithValue does this right? Original implementation uses 'nbatch'
        #self.model = step_model = policy(nenvs, 1)
        #self.model2 = train_model = policy(nbatch, nsteps)
        train_model = PolicyWithValue(ac_space, policy, value_network=None, estimate_q=False)

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        self.kfac_clip = kfac_clip

        self.is_async = is_async
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps

        # TODO: Learning rate schedule and definition of optimizer
        #self.lrschedule = lrschedule
        lrschedule = LinearTimeDecay(initial_learning_rate=lr) # TODO
        self.optim = kfac.KfacOptimizer(learning_rate=lrschedule, clip_kl=self.kfac_clip, \
                                        momentum=0.9, kfac_update=1, epsilon=0.01, \
                                        stats_decay=0.99, is_async=self.is_async, cold_iter=10,
                                        max_grad_norm=self.max_grad_norm)


        self.train_model = train_model
        #self.step_model = step_model
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state

    @tf.function
    def train(self, obs, states, rewards, masks, actions, values):
        advs = rewards - values

        with tf.GradientTape() as tape:
            # TODO: explicit watching gradients with tape.watch(x)?
            # for step in range(obs.shape[0]):
            #     cur_lr = self.lr.value()

            # TODO: Like A2C?
            policy_latent = self.train_model.policy_network(obs)
            pd, pi = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)

            #neglogpac = self.train_model.pd.neglogp(actions)
            self.logits = pi

            # TODO: states and masks like in the original implementation?
            pg_loss = tf.reduce_mean(advs * neglogpac)
            entropy = tf.reduce_mean(pd.entropy())
            pg_loss = pg_loss - self.ent_coef * entropy
            vpred = self.train_model.value(obs)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))

            #vf = tf.squeeze(self.train_model.value_fc(value_latent), axis=1)
            #vf_loss = tf.losses.mean_squared_error(tf.squeeze(self.train_model.vf), rewards)

            train_loss = pg_loss + self.vf_coef*vf_loss

            ##Fisher loss construction
            # TODO: This (up to def of params) also in 'with tf.GradientTape()'?
            self.pg_fisher = pg_fisher_loss = -tf.math.reduce_mean(neglogpac)
            sample_net = self.train_model.vf + tf.random_normal(tf.shape(self.train_model.vf))
            self.vf_fisher = vf_fisher_loss = - self.vf_fisher_coef * tf.math.reduce_mean(
                tf.pow(self.train_model.vf - tf.stop_gradient(sample_net), 2))
            self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params = params = tape.watched_variables() #Old: find_trainable_variables("acktr_model")
        # TODO: Alternativ? --> self.params = params = self.train_model.trainable_variables

        self.grads_check = grads = tape.gradient(train_loss, params) # tf.gradients(train_loss, params)
        grads_and_params = list(zip(grads, params))

        with tf.device('/gpu:0'):
            # TODO:
            #   Declaration of the optimizer now in init --> Completed
            #   cur_lr, states, advs, actions, rewards as inputs
            #self.optim = optim = kfac.KfacOptimizer(learning_rate=cur_lr, clip_kl=self.kfac_clip, \
            #                                        momentum=0.9, kfac_update=1, epsilon=0.01, \
            #                                        stats_decay=0.99, is_async=self.is_async, cold_iter=10,
            #                                        max_grad_norm=self.max_grad_norm)

            # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            self.optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = self.optim.apply_gradients(grads_and_params)

        self.q_runner = q_runner
        #self.lr = self.lrschedule() # Old
        #self.lr = Scheduler(v=self.lr, nvalues=self.total_timesteps, schedule=self.lrschedule) # Old

        return pg_loss, vf_loss, entropy


def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
          ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True, **network_kwargs):
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        policy = policy_network_fn(ob_space.shape)

    model = Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                  lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip, lrschedule=lrschedule, is_async=is_async)

    if load_path is not None:
        model.load(load_path)

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    if is_async:
        # TODO
        enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)
    else:
        enqueue_threads = []

    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    return model

