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
from collections import deque
import random
from donkeycar.parts.keras import KerasPilot


class OU(object):
    def function(self, x ,mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def save(self, img, measurement, action, reward, next_img, next_measurement, done):
        self.memory.append((img, measurement, action, reward, next_img, next_measurement, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DDPG(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args, **kwargs):
        super(DDPG, self).__init__(*args, **kwargs)

        self.actor_img, self.actor_speed, self.actor = default_model(num_action, input_shape, actor_critic='actor')
        _, _, self.actor_target = default_model(num_action, input_shape, actor_critic='actor')
        self.critic_img, self.critic_speed, self.critic_action, self.critic = default_model(num_action,input_shape,actor_critic='critic')
        _, _, _, self.critic_target = default_model(num_action, input_shape, actor_critic='critic')

        self.actor_critic_grad = tf.placeholder(tf.float32,[None, 2]) # where we will feed de/dC (from critic)
        actor_model_weights = self.actor.trainable_weights
        self.actor_grads = tf.gradients(self.actor.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)

        self.lr = 0.002
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.critic_grads = tf.gradients(self.critic.output,
                                         self.critic_action)  # where we calcaulte de/dC for feeding above
        self.epsilon = 1
        self.epsilon_decay = -5000
        self.sess = tf.Session()
        self.ou = OU()
        self.n = 0
        self.model_path = model_path
        self.batch_size = batch_size
        self.train_step = 0
        self.r_sum = 0
        self.last_state = None
        self.last_actions = None

        self.tau = 1e-3
        self.gamma = 0.99
        K.set_session(self.sess)
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())
        self.memory= Memory(500000)

        self.actor.summary()
        self.critic.summary()

        if training:
            self.compile()

    def save(self):
        self.actor.save('{}_actor.h5'.format(self.model_path))
        self.critic.save('{}_critic.h5'.format(self.model_path))
        self.actor_target.save('{}_actor_target.h5'.format(self.model_path))
        self.critic_target.save('{}_critic_target.h5'.format(self.model_path))

    def load(self, model_paths):
        '''
        :param model_paths:
        model[0] = actor
        model[1] = actor_target
        model[2] = critic
        model[3] = critic_target
        :return:
        '''
        self.actor = keras.models.load_model(model_paths[0],compile=False)
        self.actor_target = keras.models.load_model(model_paths[1],compile=False)
        self.critic = keras.models.load_model(model_paths[2], compile=False)
        self.critic_target = keras.models.load_model(model_paths[3], compile=False)

    def load_weights(self, model_paths,by_name=True):
        self.actor = keras.models.load_weights(model_paths[0], by_name=by_name)
        self.actor_target = keras.models.load_weights(model_paths[1], by_name=by_name)
        self.critic = keras.models.load_weights(model_paths[2], by_name=by_name)
        self.critic_target = keras.models.load_weights(model_paths[3], by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        self.critic.compile(optimizer=Adam(lr=self.lr),loss='mse')
        self.actor.compile(loss="mse", optimizer=Adam(lr=self.lr))

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

        target_actions = self.actor_target.predict([next_imgs, next_speeds])
        target_q = self.critic_target.predict([next_imgs, next_speeds, target_actions])
        rewards += self.gamma * target_q *(1-dones)

        evaluation = self.critic.fit([imgs, speeds, actions], rewards, verbose=0)

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

        predicted_actions = self.actor.predict([imgs, speeds])

        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_img: imgs,
            self.critic_speed: speeds,
            self.critic_action: predicted_actions,
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_img: imgs,
            self.actor_speed: speeds,
            self.actor_critic_grad: grads,
        })

    def train(self):
        batches = self.memory.sample(batch_size=self.batch_size)
        self.train_critic(batches)
        self.train_actor(batches)

        a_w, a_t_w = self.actor.get_weights(), self.actor_target.get_weights()

        # actor transfer weights
        for i in range(len(a_w)):
            a_t_w[i] = self.tau * a_w[i] + (1 - self.tau) * a_t_w[i]
        self.actor_target.set_weights(a_t_w)

        # critic transfer weights
        c_w, c_t_w = self.critic.get_weights(), self.critic_target.get_weights()
        for i in range(len(c_w)):
            c_t_w[i] = self.tau * c_w[i] + (1 - self.tau) * c_t_w[i]
        self.critic_target.set_weights(c_t_w)

    def run(self, img, speed, meter, train_state):
        if train_state > 0:
            img = np.expand_dims(img, axis=0)
            reshaped_speed = np.reshape(speed,(1, 1))
            actions = self.actor.predict([img, reshaped_speed])
            noise_t = np.zeros(2)
            noise_t[0] = max(self.epsilon, 0) * self.ou.function(actions[0][0], 0, 0.6, 0.3) # steering
            noise_t[1] = max(self.epsilon, 0) * self.ou.function(actions[0][1], 0.5, 1, 0.15) # throttle
            a_t = [(actions[0][0] + noise_t[0]), (actions[0][1] + noise_t[1])] # steering, throttle
            self.epsilon -= 1.0 / self.epsilon_decay

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

                    self.memory.save(self.last_state['img'], self.last_state['speed'], self.last_actions, reward, img, speed, done)

                    if done:
                        print("=======EPISODE DONE======")
                        print('reward sum: {}'.format(self.r_sum))
                        self.r_sum = 0
                        self.train_step = 0
                        self.last_state = None
                        self.last_actions = None

                        return 0, 0, True

                self.last_state = {
                        'img': img,
                        'speed': speed,
                        }
                self.last_actions = a_t
                self.train_step += 1

            elif train_state == 4:
                if self.n > self.batch_size:
                    print("TRAIN START!")
                    self.train()
                    print("TRAIN DONE!")
                    self.save()
                    print("SAVE DONE!")
                return 0, 0, False

            return a_t[0], a_t[1], False
        return 0, 0, False


def default_model(num_action, input_shape, actor_critic='actor'):
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Lambda, Concatenate
    LR = 1e-4  # Lower lr stabilises training greatly
    img_in = Input(shape=input_shape, name='img_in')
    if actor_critic == "actor":
        # Perception
        x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(img_in)
        #x = BatchNormalization()(x)
        x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Flatten(name='flattened')(x)
        s_in = Input(shape=(1,), name='speed')

        # speed layer
        s = Dense(64)(s_in)
        #s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        #s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        #action layer
        o = Concatenate(axis=1)([x, s])
        o = Dense(64)(o)
        #o = BatchNormalization()(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        o = Dense(num_action)(o)

        model = Model(inputs=[img_in, s_in],
                      outputs=o)
        
        # action, action_matrix, prediction from trial_run
        # reward is a function( angle, throttle)
        return img_in, s_in, model

    if actor_critic == 'critic':
        # Perception
        x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(img_in)
        x = BatchNormalization()(x)
        x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten(name='flattened')(x)
        s_in = Input(shape=(1,), name='speed')
        a_in = Input(shape=(2,), name='actions')

        # speed layer
        s = Dense(64)(s_in)
        s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        # actions_layer
        a = Dense(64)(a_in)
        a = BatchNormalization()(a)
        a = Dropout(0.5)(a)
        a = Activation('relu')(a)
        a = Dense(32)(a)
        a = BatchNormalization()(a)
        a = Dropout(0.5)(a)
        a = Activation('relu')(a)

        o = Concatenate(axis=1)([x, s, a])
        o = Dense(64)(o)
        o = BatchNormalization()(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        q = Dense(1)(o)
        model = Model(inputs=[img_in, s_in, a_in], outputs=q)

        return img_in, s_in, a_in, model
