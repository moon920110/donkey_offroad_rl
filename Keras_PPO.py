import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import utils
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
from scipy import signal


class MemoryPPO:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    This is adapted version of the Spinning Up Open Ai PPO code of the buffer.
    https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
    """

    def __init__(self, img_dim, speed_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)

        # just empty arrays with appropriate sizes
        self.img_buf = np.zeros(combined_shape(
            size, img_dim), dtype=np.float32)  # imgs
        self.speed_buf = np.zeros(combined_shape(
            size, speed_dim), dtype=np.float32)  # speeds
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

    def store(self, img, speed, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.img_buf[self.ptr] = img
        self.speed_buf[self.ptr] = speed
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Finishes an episode of data collection by calculating the diffrent rewards and resetting pointers.
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
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam) 

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.img_buf[pos_lst], self.speed_buf[pos_lst], self.act_buf[pos_lst], self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def get_log_probability_density(pred, y):
        mu_and_sigma = pred
        mu = mu_and_sigma[:, :2]
        sigma = mu_and_sigma[:, 2:]
        variance = K.square(sigma)
        pdf = 1. / K.sqrt(2. * np.pi * variance) * K.exp(-K.square(y - mu) / (2. * variance))
        log_pdf = K.log(pdf + K.epsilon())

        return log_pdf

    def loss(y_true, y_pred):
        PPO_LOSS_CLIPPING = 0.2
        PPO_ENTROPY_LOSS = 5 * 1e-3  # Does not converge without entropy penalty

        log_pdf_new = get_log_probability_density(y_pred, y_true)
        log_pdf_old = get_log_probability_density(old_prediction, y_true)

        ratio = K.exp(log_pdf_new - log_pdf_old)
        surrogate1 = ratio * advantage
        clip_ratio = K.clip(ratio, min_value=(1-PPO_LOSS_CLIPPING), max_value=(1+PPO_LOSS_CLIPPING))
        surrogate2 = clip_ratio * advantage

        loss_actor = -K.mean(K.minimum(surrogate1, surrogate2))

        sigma = y_pred[:, 2:]
        variance = K.square(sigma)

        loss_entropy = PPO_ENTROPY_LOSS * K.mean(-(K.log(2*np.pi*variance)+1) / 2)

        return loss_actor+loss_entropy

    return loss


class PPO(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

        self.actor = default_model(num_action, input_shape, actor_critic='actor')
        self.old_actor = default_model(num_action, input_shape, actor_critic='actor')
        self.critic = default_model(num_action, input_shape,actor_critic='critic')
        self.old_actor.set_weights(self.actor.get_weights())

        self.num_action = num_action
        self.n = 0
        self.gamma = 0.99
        self.lmbda = 0.95
        self.lr = 0.002
        self.eps = 0.1
        self.K = 2
        self.batch_size = batch_size
        self.model_path = model_path
        self.optimal = False

        self.r_sum = 0
        self.last_state = None
        self.last_action = None
        self.train_step = 0

        self.memory = MemoryPPO(img_dim=input_shape, speed_dim=(1,), act_dim=num_action, size=self.batch_size, gamma=0.99, lam=0.95)
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2*num_action))

        self.actor.summary()
        self.critic.summary()

        if training:
            self.compile()

    def save(self):
        self.critic.save('{}_critic.h5'.format(self.model_path))
        self.actor.save('{}_actor.h5'.format(self.model_path))

    def load(self, model_paths):
        '''
        :param model_paths:
        model[0] = actor
        model[1] = critic
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

    def train(self):
        imgs, speeds, actions, gae_advantages, rewards, values = self.memory.get_batch(self.batch_size)
        gae_advantages = gae_advantages.reshape(-1, 1)  # batches of shape (1,) required
        gae_advantages = utils.normalize(gae_advantages)  # optionally normalize

        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(imgs, speeds)

        # commit training
        self.actor.fit(
            x=[imgs, speeds, gae_advantages, batch_old_prediction], y=actions, verbose=0)
        self.critic.fit(
            x=[imgs, speeds], y=rewards, epochs=1, verbose=0)
        # update old network
        alpha = 0.9
        actor_weights = np.array(self.actor.get_weights())
        actor_tartget_weights = np.array(self.old_actor.get_weights())
        new_weights = alpha * actor_weights + (1 - alpha) * actor_tartget_weights
        self.old_actor.set_weights(new_weights)

    def run(self, img, speed, meter, train_state):
        if train_state > 0:
            reshaped_img = np.expand_dims(img, axis=0)
            reshaped_speed = np.expand_dims(speed, axis=0)
            mu_and_sigma = self.actor.predict([reshaped_img, reshaped_speed, self.dummy_advantage, self.dummy_old_prediciton])
            mu = mu_and_sigma[0, 0:self.num_action]
            sigma = mu_and_sigma[0, self.num_action:]

            if self.optimal:
                action = mu
            else:
                action = np.random.normal(loc=mu, scale=sigma, size=2)

            if train_state < 4:
                if self.train_step > 0:
                    self.n += 1

                    reward = meter - self.train_step * 0.01
                    if train_state == 2:
                        reward -= 100
                    elif train_state == 3:
                        reward += 100
                    done = train_state > 1
                    self.r_sum += reward
                    
                    last_img = self.last_state['img']
                    last_speed = self.last_state['speed']
                    v = self._get_v(np.expand_dims(last_img, axis=0), np.expand_dims(last_speed, axis=0))
                    self.memory.store(last_img, last_speed, self.last_actions, reward ,v)

                    if done or (self.train_step % self.batch_size == 0):
                        print('reward sum: {}'.format(self.r_sum))
                        final_val = reward if done else self._get_v(reshaped_img, reshaped_speed)
                        self.memory.finish_path(final_val)

                        if done:
                            print("======EPISODE DONE======")
                            self.r_sum = 0
                            self.train_step = 0
                            self.last_state = None
                            self.last_actions = None
                        else:
                            self.last_state = {
                                        'img': img,
                                        'speed': speed,
                                        }
                            self.last_actions = action

                            self.train_step += 1

                        return 0, 0, True

                self.last_state = {
                        'img': img,
                        'speed': speed,
                        }
                self.last_actions = action
                self.train_step += 1

            elif train_state == 4:
                print("TRAIN START!")
                self.train()
                print("TRAIN DONE!")
                self.save()  # It is fixed by editing tensorflow_core code
                print("SAVE DONE!")
                self.memory.clear()
                print("MEMORY CLEAR!")
                return 0, 0, False
                
            return action[0], action[1], False
        return 0, 0, False

    def _get_v(self, img, speed):
        v = self.critic.predict([img, speed])#.flatten()
        return v

    def get_old_prediction(self, imgs, speeds):
        return self.old_actor.predict_on_batch([imgs, speeds, self.dummy_advantage, self.dummy_old_prediciton])


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
        adv_in = Input(shape=(1,), name='adv')
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
        mu = Dense(num_action, activation='tanh',name="actor_output_mu")(o)
        sigma = Dense(num_action, activation='softplus', name="actor_output_sigma")(o)
        mu_and_sigma = Concatenate(axis=-1)([mu, sigma])

        model = Model(inputs=[img_in, s_in, adv_in, old_prediction],
                      outputs=mu_and_sigma)
        model.compile(optimizer=Adam(lr=0.002),
<<<<<<< HEAD
                      loss=proximal_policy_optimization_loss_continuous(advantage=adv_in,old_prediction=old_prediction))
=======
                      loss=proximal_policy_optimization_loss_continuous(advantage=adv_in,
                                                                         old_prediction=old_prediction))
>>>>>>> 6beb3688d93df1213ad8f76030adb8054b0b89e8
        # action, action_matrix, prediction from trial_run
        # reward is a function( angle, throttle)
        return model

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

        return model
