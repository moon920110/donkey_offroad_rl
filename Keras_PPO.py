import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose
import numpy as np
from tensorflow.python.keras.optimizers import Adam
# from keras import KerasPilot
from collections import deque
import random
from donkeycar.parts.keras import KerasPilot
import tensorflow_probability as tfp
tfd = tfp.distributions

class MemoryPPO:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    This is adapted version of the Spinning Up Open Ai PPO code of the buffer.
    https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)
        # just empty arrays with appropriate sizes
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)  # states
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)  # actions
        # actual rwards from state using action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # predicted values of state
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # gae advantewages
        self.ret_buf = np.zeros(size, dtype=np.float32)  # discounted rewards
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        example input: [x0, x1, x2] output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Finishes an episode of data collection by calculating the diffrent rewards and resetting pointers.
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.obs_buf[pos_lst], self.act_buf[pos_lst], self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0

def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        mean_sq_err = K.mean(K.square(y_pred - y_true), axis=-1)
        try:
            PPO_OUT_OF_RANGE = 1000  # negative of -1000
            checkifzero = K.sum(old_prediction, PPO_OUT_OF_RANGE)
            divbyzero = old_prediction / checkifzero
        except:
            return mean_sq_err
        PPO_NOISE = 1.0
        var = keras.backend.square(PPO_NOISE)
        denom = K.sqrt(2 * np.pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))
        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)
        PPO_LOSS_CLIPPING = 0.2
        PPO_ENTROPY_LOSS = 5 * 1e-3  # Does not converge without entropy penalty

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING,
                                                       max_value=1 + PPO_LOSS_CLIPPING) * advantage)) + PPO_ENTROPY_LOSS * (
                           prob * K.log(prob + 1e-10))



    return loss
class PPO(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

        self.actor_img, self.actor_speed, self.actor = default_model(num_action, input_shape, actor_critic='actor')
        self.old_actor_img, self.old_actor_speed, self.old_actor = default_model(num_action, input_shape, actor_critic='actor')
        self.critic_img, self.critic_speed,self.critic_action, self.critic = default_model(num_action, input_shape,actor_critic='critic')
        self.old_actor.set_weights(self.actor.get_weights())
        self.n = 0
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 2
        self.memory = MemoryPPO(obs_dim=(84,84,3),act_dim=2,size=50000,gamma=0.99, lam=0.95)
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 4))
    def load(self, model_paths):
        '''
        :param model_paths:
        model[0] = actor
        model[1] = actor_target
        model[2] = critic
        model[3] = critic_target
        :return:
        '''
        self.actor = keras.models.load_model(model_paths[0], compile=False)
        self.critic = keras.models.load_model(model_paths[1], compile=False)

    def load_weights(self, model_paths, by_name=True):
        self.actor = keras.models.load_weights(model_paths[0], by_name=by_name)
        self.critic = keras.models.load_weights(model_paths[1], by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        self.critic.compile(optimizer=Adam(lr=self.lr), loss={'total_reward': 'mean_squared_error'})

    def train(self, batch_size):
        img,speeds, actions, gae_advantages, rewards, values = self.memory.sample(batch_size)
        batches = self.memory.sample(batch_size=batch_size)
        gae_advantages = gae_advantages.reshape(-1, 1)  # batches of shape (1,) required
        gae_advantages = K.utils.normalize(gae_advantages)  # optionally normalize
        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(img,speeds)
        # commit training
        self.actor_network.fit(
            x=[states, gae_advantages, batch_old_prediction], y=actions, verbose=0)
        self.critic_network.fit(
            x=states, y=rewards, epochs=1, verbose=0)
        # update old network
        alpha = 0.9
        actor_weights = np.array(self.actor.get_weights())
        actor_tartget_weights = np.array(self.old_actor.get_weights())
        new_weights = alpha * actor_weights + (1 - alpha) * actor_tartget_weights
        self.old_actor.set_weights(new_weights)

    def run(self, img, speed,optimal=False):
        mu_and_sigma = self.actor_network.predict([img,speed, self.dummy_advantage, self.dummy_old_prediciton])
        mu = mu_and_sigma[0, 0:2]
        sigma = mu_and_sigma[0, 2:]
        if optimal:
            action = mu
        else:
            action = np.random.normal(loc=mu, scale=sigma, size=2)
        return action
    def add(self,img,speed):
        v = self.critic.predict([img,speed]).flatten()
        self.memory.add(v)
    def get_old_prediction(self,imgs,speeds):
        return self.old_actor.predict_on_batch([imgs,speeds, self.dummy_advantage, self.dummy_old_prediciton])
def default_model(num_action, input_shape, actor_critic='actor'):
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Lambda, Concatenate
    LR = 1e-4  # Lower lr stabilises training greatly
    img_in = Input(shape=input_shape, name='img_in')
    if actor_critic == "actor":
        # Perception

        x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(img_in)
        x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        s_in = Input(shape=(1,), name='speed')
        adv_in = Input(shape=(1,))
        old_prediction = Input(shape=(2*num_action,), name='old_prediction_input')
        # speed layer
        s = Dense(64)(s_in)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        # action layer
        o = Concatenate(axis=1)([x, s])
        o = Dense(64)(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        mu = Dense(2, activation='tanh',name="actor_output_mu")(o)
        sigma = Dense(2, activation='softplus', name="actor_output_sigma")(o)
        mu_and_sigma = Concatenate([mu, sigma])
        model = Model(inputs=[img_in, s_in,adv_in,old_prediction],
                      outputs=mu_and_sigma)
        model.compile(optimizer=Adam(lr=0.002),
                      loss=[proximal_policy_optimization_loss_continuous(advantage=adv_in,
                                                                         old_prediction=old_steer_in),
                            proximal_policy_optimization_loss_continuous(advantage=adv_in,
                                                                         old_prediction=old_throttle_in)]
        # action, action_matrix, prediction from trial_run
        # reward is a function( angle, throttle)
        return img_in, s_in, model

    if actor_critic == 'critic':
        # Perception
        x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(img_in)
        x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        s_in = Input(shape=(1,), name='speed')

        # speed layer
        s = Dense(64)(s_in)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        o = Concatenate(axis=1)([x, s])
        o = Dense(64)(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        total_reward = Dense(units=1, activation='linear', name='total_reward')(o)
        model = Model(inputs=[img_in, s_in], outputs=total_reward)

        return img_in, s_in, model
