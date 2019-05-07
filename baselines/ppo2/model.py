import tensorflow as tf
from baselines.common.policies import PolicyWithValue

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, ac_space, policy_network, value_network=None, ent_coef, vf_coef, max_grad_norm, lr):
        self.train_model = PolicyWithValue(ac_space, policy_network, False)
        if MPI is not None:
          self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=lr, epsilon=1e-5)
        else:
          self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-5)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

    @tf.function
    def train(self, cliprange, obs, returns, masks, actions, values, neglogpac_old, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs)
            vpredclipped = values + tf.clip_by_value(vpred - values, -cliprange, cliprange)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            ratio = tf.exp(neglogpac_old - neglogpac)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - neglogpac_old))
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))

            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy, approxkl, clipfrac, vpred, vpredclipped
