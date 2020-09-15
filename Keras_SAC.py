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

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def save(self, img, measurement, action, reward, next_img, next_measurement, done):
        self.memory.append((img, measurement, action, reward, next_img, next_measurement, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class SAC(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3),steer_range=[-0.8,0.8],throttle_range=[-1.0,1.0], *args, **kwargs):
        super(SAC, self).__init__(*args, **kwargs)

        self.steer_scale = tf.tenseor((steer_range[1] - steer_range[0]) / 2)
        self.steer_bias = tf.tenseor((steer_range[1] + steer_range[0]) / 2)
        self.throttle_scale = tf.tenseor((throttle_range[1] - throttle_range[0]) / 2)
        self.throttle_bias = tf.tenseor((throttle_range[1] + throttle_range[0]) / 2)

        self.actor_img, self.actor_speed, self.actor = default_model(num_action, input_shape, actor_critic='actor')
        _, _, self.actor_target = default_model(num_action, input_shape, actor_critic='actor')
        self.critic_img1, self.critic_speed1,self.critic_action1, self.critic1 = default_model(num_action, input_shape,actor_critic='critic')
        self.critic_img2, self.critic_speed2,self.critic_action2, self.critic2 = default_model(num_action, input_shape, actor_critic='critic')
        _, _, self.critic_target1 = default_model(num_action, input_shape, actor_critic='critic')
        _, _, self.critic_target2 = default_model(num_action, input_shape, actor_critic='critic')
        self.alpha = tf.Variable(0.0, dtype=tf.float64)
        self.target_entropy = -tf.constant(2, dtype=tf.float64)
        self.n = 0
        self.gamma = 0.99
        self.memory = Memory(50000)
        self.train_step = 0
        self.last_state = None
        self.last_actions = None
    def save(self):
        self.actor.save('{}_actor.h5'.format(self.model_path))
        self.critic.save('{}_critic.h5'.format(self.model_path))
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
        self.actor.compile(loss=actor_loss, optimizer=Adam(lr=self.lr))
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)
    # Define actor_loss
    def actor_loss():

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            return K.mean(y_pred - y_true, axis=-1)

        # Return a function
        return loss
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

        q1 =self.critic1([imgs,speeds,actions])
        q2 = self.critic2([imgs, speeds,actions])
        pi,log_pi = self.actor.predict([imgs,speeds])
        tq1 = self.critic_target1.predict([next_imgs,log_pi])
        tq2 = self.critic_target2.predict([next_imgs, log_pi])
        tq = tf.minimum(tq1, tq2)
        soft_tq = tq - self.alpha * log_pi
        y = tf.stop_gradient(rewards + self.gamma * dones * soft_tq)
        self.critic1.fit(q1,y)
        self.critic2.fit(q2,y)
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
        pi,log_pi = self.actor.predict([imgs,speeds])
        q1 = self.critic1.predict([imgs, speeds, actions])
        q2 = self.critic2.predict([imgs, speeds, actions])
        tq = tf.minimum(tq1, tq2)
        self.actor.fit(tq,self.alpha * log_pi)
    def update_alpha(self, batches):
        batches = np.array(batches).transpose()
        imgs = np.vstack(batches[0])
        speeds = np.vstack(batches[1])
        actions = np.vstack(batches[2])
        rewards = np.vstack(batches[3])
        next_imgs = np.vstack(batches[4])
        next_speeds = np.vstack(batches[5])
        dones = np.vstack(batches[6].astype(int))
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi, log_pi = self.actor([imgs,speeds])

            alpha_loss = tf.reduce_mean( - self.alpha*(log_pi + self.target_entropy))

        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
    def train(self, batch_size):
        batches = self.memory.sample(batch_size=batch_size)
        self.train_critic(batches)
        self.train_actor(batches)
        self.update_alpha(batches)
        # update weight
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

    def run(self, img, speed,meter,train_state):
        speed = np.reshape(speed, (1, 1))
        # run by Gaussian Policy (other case Deterministic Policy)
        mu,log_std = self.actor.predict([img, speed])

        ste_std = tf.exp(log_std[0])
        ste_normal = tfd.Normal(mu[0],std[0])
        thr_std = tf.exp(log_std[1])
        thr_normal = tfd.Normal(mu[1], std[1])
        sx_t = ste_normal.sample()  # for reparameterization trick (mean + std * N(0,1))
        tx_t = thr_normal.sample()
        sy_t = tf.tanh(sx_t)
        ty_t = tf.tanh(tx_t)
        steer = sy_t * self.steer_scale + self.steer_bias
        throttle = ty_t * self.throttle_scale + self.throttle_bias
        ste_log_prob = ste_normal.log_prob(sx_t)
        thr_log_prob = thr_normal.log_prob(tx_t)
        # Enforcing Action Bound
        ste_log_prob -= tf.log(self.steer_scale * (1 - sy_t.pow(2)) + 1e-6)
        thr_log_prob -= tf.log(self.throttle_scale * (1 - ty_t.pow(2)) + 1e-6)
        ste_log_prob = ste_log_prob.sum(1, keepdim=True)
        thr_log_prob = thr_log_prob.sum(1, keepdim=True)
        ste_mu = tf.tanh(mu[0]) * self.steer_scale + self.steer_bias
        thr_mu =tf.tanh(mu[1]) * self.action_scale + self.action_bias
        #return [steer,throttle], [ste_log_prob,thr_log_prob], [ste_mu,thr_mu]
        if train_state < 4:
            if self.train_step > 0 :
                self.n += 1
                reward = mete - self.train_step * 0.01
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

                    return 0, 0

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
            return 0, 0

        return steer, throttle
    return 0, 0

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
        mu = Dropout(0.5)(mu)
        mu = Activation('tanh')(mu)
        std = Dense(num_action)(o)
        std = Dropout(0.5)(std)
        std = Activation('tanh')(std)
        log _std = Lambda(lambda x: tf.clip_by_value(x,min=-20,max=2))(std)

        model = Model(inputs=[img_in, s_in],
                      outputs=[mu,log_std])

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
        o = Dense(1)(o)
        o = Dropout(0.5)(o)
        q = Activation('relu')(o)
        model = Model(inputs=[img_in, s_in, a_in], outputs=q)

        return img_in, s_in, a_in, model
