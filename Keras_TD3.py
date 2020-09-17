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



class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def save(self, img, measurement, action, reward, next_img, next_measurement, done):
        self.memory.append((img, measurement, action, reward, next_img, next_measurement, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class TD3(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args,
                 **kwargs):
        super(TD3, self).__init__(*args, **kwargs)

        self.actor_img, self.actor_speed, self.actor = default_model(num_action, input_shape, actor_critic='actor')
        _, _, self.actor_target = default_model(num_action, input_shape, actor_critic='actor')
        self.critic_img1, self.critic_speed1, self.critic_action1, self.critic1 = default_model(num_action, input_shape,
                                                                                                actor_critic='critic')
        self.critic_img2, self.critic_speed2, self.critic_action2, self.critic2 = default_model(num_action, input_shape,
                                                                                                actor_critic='critic')
        _, _, self.critic_target1 = default_model(num_action, input_shape, actor_critic='critic')
        _, _, self.critic_target2 = default_model(num_action, input_shape, actor_critic='critic')

        self.lr = 0.002
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        self.n = 0
        self.model_path = model_path
        self.batch_size = batch_size
        self.train_step = 0
        self.r_sum = 0
        self.last_state = None
        self.last_actions = None

        self.tau = 1e-3
        self.gamma = 0.99
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
        # Initialize for later gradient calculations
        self.memory = Memory(5000)

        self.actor.summary()
        self.critic1.summary()
        self.critic2.summary()
        if training:
            self.compile()

    def save(self):
        self.actor.save('{}_actor.h5'.format(self.model_path))
        self.critic1.save('{}_critic.h5'.format(self.model_path))
        self.critic2.save('{}_critic.h5'.format(self.model_path))
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
        self.critic1.compile(optimizer=Adam(lr=self.lr), loss='mse')
        self.critic2.compile(optimizer=Adam(lr=self.lr), loss='mse')
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

        noise = np.clip(np.random.randn(2) * self.policy_noise, -self.noise_clip, self.noise_clip)
        target_actions = self.actor_target([next_imgs,next_speeds]) + noise
        target_actions = K.clip(target_actions,[-0.8,0],[0.8,1])

        target_q1 = self.critic_target1.predict([next_imgs, next_speeds, target_actions],steps=1)
        target_q2 = self.critic_target2.predict([next_imgs, next_speeds, target_actions],steps=1)
        target_q = K.minimum(target_q1,target_q2)
        rewards += self.gamma * target_q * (1 - dones)
        with tf.GradientTape() as tape:
            q1 = self.critic1([imgs,speeds,actions])
            q2 = self.critic2([imgs, speeds, actions])
            loss1 = tf.reduce_mean(tf.keras.losses.mean_squared_error(rewards, q1))
            loss2 = tf.reduce_mean(tf.keras.losses.mean_squared_error(rewards, q2))
            loss = loss1 + loss2
        grads = tape.gradient(loss, self.critic1.trainable_weights + self.critic2.trainable_weights)
        self.critic1.optimizer.apply_gradients(
            zip(grads, self.critic1.trainable_weights + self.critic2.trainable_weights))

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

        with tf.GradientTape() as tape:
            actions = self.actor([imgs,speeds])
            actions = tf.clip_by_value(actions, [-0.8,0],[0.8,1])
            q = self.critic1([state, actions])
            loss = -tf.reduce_mean(q)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))


    def train(self):
        batches = self.memory.sample(batch_size=self.batch_size)
        self.train_critic(batches)
        self.train_actor(batches)

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
            reshaped_speed = np.reshape(speed, (1, 1))
            actions = self.actor.predict([img, reshaped_speed])[0]

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

                    self.memory.save(self.last_state['img'], self.last_state['speed'], self.last_actions, reward, img,
                                     speed, done)

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
                self.last_actions = actions
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

        # speed layer
        s = Dense(64)(s_in)
        s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)
        s = Dense(64)(s)
        s = BatchNormalization()(s)
        s = Dropout(0.5)(s)
        s = Activation('relu')(s)

        # action layer
        o = Concatenate(axis=1)([x, s])
        o = Dense(64)(o)
        o = BatchNormalization()(o)
        o = Dropout(0.5)(o)
        o = Activation('relu')(o)
        o = Dense(num_action)(o)
        o = Activation('linear')(o)

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
