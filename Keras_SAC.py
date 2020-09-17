import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
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
from collections import deque
import random
from donkeycar.parts.keras import KerasPilot
import tensorflow_probability as tfp

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def save(self, img, measurement, action, reward, next_img, next_measurement, done):
        self.memory.append((img, measurement, action, reward, next_img, next_measurement, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class SAC(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args, **kwargs):
        super(SAC, self).__init__(*args, **kwargs)

        steer_range = [-1., 1.]
        throttle_range = [0., 1.]

        self.steer_scale = ops.convert_to_tensor((steer_range[1] - steer_range[0]) / 2)
        self.steer_bias = ops.convert_to_tensor((steer_range[1] + steer_range[0]) / 2)
        self.throttle_scale = ops.convert_to_tensor((throttle_range[1] - throttle_range[0]) / 2)
        self.throttle_bias = ops.convert_to_tensor((throttle_range[1] + throttle_range[0]) / 2)

        self.actor = default_model(num_action, input_shape, actor_critic='actor')
        self.actor_target = default_model(num_action, input_shape, actor_critic='actor')
        self.critic1 = default_model(num_action, input_shape,actor_critic='critic')
        self.critic2 = default_model(num_action, input_shape, actor_critic='critic')
        self.critic_target1 = default_model(num_action, input_shape, actor_critic='critic')
        self.critic_target2 = default_model(num_action, input_shape, actor_critic='critic')

        self.model_path = model_path
        self.batch_size = batch_size
        self.alpha = tf.Variable(0.0, dtype=tf.float32)
        self.target_entropy = -tf.constant(2, dtype=tf.float32)
        self.n = 0
        self.lr = 1e-4
        self.gamma = 0.99
        self.tau = 1e-3
        self.memory = Memory(50000)
        self.train_step = 0
        self.last_state = None
        self.last_actions = None

        if training:
            self.critic1_optimizer = tf.keras.optimizers.Adam(self.lr)
            self.critic2_optimizer = tf.keras.optimizers.Adam(self.lr)
            self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
            self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)

    def save(self):
        self.actor.save('{}_actor.h5'.format(self.model_path))
        self.critic1.save('{}_critic1.h5'.format(self.model_path))
        self.critic2.save('{}_critic2.h5'.format(self.model_path))
        self.actor_target.save('{}_actor_target.h5'.format(self.model_path))
        self.critic_target1.save('{}_critic_target1.h5'.format(self.model_path))
        self.critic_target2.save('{}_critic_target2.h5'.format(self.model_path))

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
        self.actor_target = keras.models.load_model(model_paths[1], compile=False)
        self.critic1 = keras.models.load_model(model_paths[2], compile=False)
        self.critic_target1 = keras.models.load_model(model_paths[3], compile=False)
        self.critic2 = keras.models.load_model(model_paths[4], compile=False)
        self.critic_target2 = keras.models.load_model(model_paths[5], compile=False)

    def load_weights(self, model_paths, by_name=True):
        self.actor = keras.models.load_weights(model_paths[0], by_name=by_name)
        self.actor_target = keras.models.load_weights(model_paths[1], by_name=by_name)
        self.critic1 = keras.models.load_weights(model_paths[2], by_name=by_name)
        self.critic_target1 = keras.models.load_weights(model_paths[3], by_name=by_name)
        self.critic2 = keras.models.load_weights(model_paths[4], by_name=by_name)
        self.critic_target2 = keras.models.load_weights(model_paths[5], by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def train_critic(self, batches):
        batches = np.array(batches).transpose()

        imgs = np.vstack(batches[0])
        speeds = np.vstack(batches[1])
        actions = np.vstack(batches[2])
        rewards = np.vstack(batches[3])
        next_imgs = np.vstack(batches[4])
        next_speeds = np.vstack(batches[5])
        dones = np.vstack(batches[6].astype(int))

        speeds = np.reshape(speeds, (-1, 1))
        next_speeds = np.reshape(next_speeds, (-1, 1))

        # critic1 update
        q1 = self.critic1([imgs, speeds, actions])
        q2 = self.critic2([imgs, speeds, actions])

        pi, log_pi = self.actor.predict([next_imgs, next_speeds])

        tq1 = self.critic_target1.predict([next_imgs, next_speeds, pi])
        tq2 = self.critic_target2.predict([next_imgs, next_speeds, pi])

        tq = tf.minimum(tq1, tq2)
        soft_tq = tq - self.alpha * log_pi
        y = tf.stop_gradient(rewards + self.gamma * dones * soft_tq)

        critic1_loss = tf.reduce_mean((q1 - y) ** 2)
        critic2_loss = tf.reduce_mean((q2 - y) ** 2)

        grads1 = tf.gradients(critic1_loss, self.critic1.trainable_variables)
        grads2 = tf.gradients(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        self.critic1_optimizer.apply_gradients(zip(grads2, self.critic2.trainable_variables))

    def train_actor(self, batches):
        batches = np.array(batches).transpose()
        imgs = np.vstack(batches[0])
        speeds = np.vstack(batches[1])
        actions = np.vstack(batches[2])
        rewards = np.vstack(batches[3])
        next_imgs = np.vstack(batches[4])
        next_speeds = np.vstack(batches[5])
        dones = np.vstack(batches[6].astype(int))

        speeds = np.reshape(speeds, (-1, 1))
        next_speeds = np.reshape(next_speeds, (-1, 1))

        pi, log_pi = self.actor([imgs, speeds])

        q1 = self.critic1.predict([imgs, speeds, actions])
        q2 = self.critic2.predict([imgs, speeds, actions])
        tq = tf.minimum(q1, q2)

        soft_q = tq - self.alpha * log_pi

        actor_loss = -tf.reduce_mean(soft_q)

        variables = self.actor.trainable_variables
        grads = tf.gradients(actor_loss, variables)
        self.actor_optimizer.apply_gradients(zip(grads, variables))

    def update_alpha(self, batches):
        batches = np.array(batches).transpose()
        imgs = np.vstack(batches[0])
        speeds = np.vstack(batches[1])
        actions = np.vstack(batches[2])
        rewards = np.vstack(batches[3])
        next_imgs = np.vstack(batches[4])
        next_speeds = np.vstack(batches[5])
        dones = np.vstack(batches[6].astype(int))

        # Sample actions from the policy for current states
        pi, log_pi = self.actor([imgs, speeds])
        alpha_loss = tf.reduce_mean(-self.alpha * (log_pi + self.target_entropy))

        variables = [self.alpha]
        grads = tf.gradients(alpha_loss, variables)

        self.alpha_optimizer.apply_gradients(zip(grads, variables))

    def train(self):
        batches = self.memory.sample(batch_size=self.batch_size)
        self.train_critic(batches)
        self.train_actor(batches)
        self.update_alpha(batches)

        # update weight
        a_w, a_t_w = self.actor.get_weights(), self.actor_target.get_weights()

        # actor transfer weights
        for i in range(len(a_w)):
            a_t_w[i] = self.tau * a_w[i] + (1 - self.tau) * a_t_w[i]
        self.actor_target.set_weights(a_t_w)

        # critic1 transfer weights
        c_w, c_t_w = self.critic1.get_weights(), self.critic_target1.get_weights()
        for i in range(len(c_w)):
            c_t_w[i] = self.tau * c_w[i] + (1 - self.tau) * c_t_w[i]
        self.critic_target1.set_weights(c_t_w)

        # critic2 transfer weights
        c_w, c_t_w = self.critic2.get_weights(), self.critic_target2.get_weights()
        for i in range(len(c_w)):
            c_t_w[i] = self.tau * c_w[i] + (1 - self.tau) * c_t_w[i]
        self.critic_target2.set_weights(c_t_w)

    def run(self, img, speed, meter, train_state):
        if train_state > 0:
            img = np.expand_dims(img, axis=0)
            speed = np.reshape(speed, (1, 1))

            # run by Gaussian Policy (other case Deterministic Policy)
            actions,_ = self.actor.predict([img, speed])

            #return [steer,throttle], [ste_log_prob,thr_log_prob], [ste_mu,thr_mu]
            if train_state < 4:
                if self.train_step > 0 :
                    self.n += 1
                    reward = meter - self.train_step * 0.01
                    if train_state == 2:
                        reward -= 100
                    elif train_state == 3:
                        reward += 100
                    done = train_state > 1
                    self.memory.save(self.last_state['img'], self.last_state['speed'], self.last_actions, reward, img, speed, done)
                    if done:
                        print("=======EPISODE DONE======")
                        print('reward: {}'.format(reward))
                        self.train_step = 0
                        self.last_state = None
                        self.last_actions = None

                        return 0, 0, True

                self.last_state = {
                    'img': img,
                    'speed': speed,
                }
                self.last_actions = actions[0]
                self.train_step += 1

            elif train_state == 4:
                if self.n > self.batch_size:
                    print("TRAIN START!")
                    self.train()
                    print("TRAIN DONE!")
                    self.save()
                    print("SAVE DONE!")
                return 0, 0, False

            return actions[:, 0], actions[:, 1], False
        return 0, 0, False


def default_model(num_action, input_shape, actor_critic='actor'):
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Lambda, Concatenate
    LR = 1e-4  # Lower lr stabilises training greatly
    img_in = Input(shape=input_shape, name='img_in')
    EPSILON = 2e-3
    if actor_critic == "actor":
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

        # action layer
        o = Concatenate(axis=1)([x, s])
        o = Dense(64)(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)

        mu = Dense(num_action)(o)
        mu = Activation('tanh')(mu)
        std = Dense(num_action)(o)
        std = Activation('tanh')(std)
        log_std = Lambda(lambda x: clip_ops.clip_by_value(x, 20, 2))(std)

        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mu, std)
        actions = dist.sample()
        actions = tf.tanh(actions)

        log_pi = dist.log_prob(actions)
        log_pi = log_pi - tf.reduce_sum(tf.math.log(1 - actions ** 2 + EPSILON), axis=1, keepdims=True)
        model = Model(inputs=[img_in, s_in], outputs=[actions, log_pi])

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
        a_in = Input(shape=(2,), name='actions')

        # speed layer
        s = Dense(64)(s_in)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        # actions_layer
        a = Dense(64)(a_in)
        a = Dropout(0.5)(a)
        a = Activation('relu')(a)
        a = Dense(32)(a)
        a = Dropout(0.5)(a)
        a = Activation('relu')(a)

        o = Concatenate(axis=1)([x, s, a])
        o = Dense(64)(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        q = Dense(1)(o)
        model = Model(inputs=[img_in, s_in, a_in], outputs=q)

        return model
