import torch.nn as nn
import torch as tr
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random

from collections import deque
from baseagent import BaseAgent


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


class DDPGAgent(BaseAgent):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args,
                 **kwargs):
        super(DDPGAgent, self).__init__(*args, **kwargs)

        self.perception = Perception()
        self.actor = Actor(num_action)
        self.actor_target = Actor(num_action)
        self.critic = Critic()
        self.critic_target = Critic()

        # update target model
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(self.critic.state_dict())
        self.tau = 1e-3

        # set optimizer
        self.lr = 0.001
        #self.decay = -5000
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr,)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr,)
        self.perc_optim = optim.Adam(self.perception.parameters(), lr=self.lr,)

        # common settings
        self.gamma = 0.99
        self.memory = Memory(50000)
        self.model_path = model_path
        self.n = 0
        self.train_step = 0
        self.r_sum = 0
        self.last_state = None
        self.last_actions = None
        self.batch_size = batch_size

        # about DDPG
        self.ou = OU()
        self.epsilon = 1
        self.epsilon_decay = -5000

    def save(self):
        tr.save(self.actor.state_dict(),'{}_actor.h5'.format(self.model_path))
        tr.save(self.actor_target.state_dict(), '{}_actor_target.h5'.format(self.model_path))
        tr.save(self.critic.state_dict(), '{}_critic.h5'.format(self.model_path))
        tr.save(self.critic_target.state_dict(), '{}_critic_target.h5'.format(self.model_path))

    def load(self,model_paths):
        self.actor = tr.load(model_paths[0])
        self.actor_target = tr.load(model_paths[1])
        self.critic = tr.load(model_paths[2])
        self.critic_target = tr.load(model_paths[3])

    def train(self):
        # sample in Memory
        batches = self.memory.sample(batch_size=self.batch_size)
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

        # convert to torch tensor
        imgs = tr.from_numpy(imgs).float()
        next_imgs = tr.from_numpy(next_imgs).float()
        speeds = tr.from_numpy(speeds).float()
        next_speeds = tr.from_numpy(next_speeds).float()
        actions = tr.from_numpy(actions).float()
        rewards = tr.from_numpy(rewards).float()
        dones = tr.from_numpy(dones).float()

        # train critic
        feature = self.perception(imgs, speeds)
        pred_q = self.critic(feature, actions)
        next_feature = self.perception(next_imgs, next_speeds)
        targ_act = self.actor_target(next_feature)
        targ_q = self.critic_target(next_feature, targ_act)
        target = rewards + (1 - dones) * self.gamma * targ_q
        c_loss = F.mse_loss(pred_q, target).mean()
        self.critic_optim.zero_grad()
        c_loss.backward()
        self.critic_optim.step()

        # train actor
        self.actor_optim.zero_grad()
        feature = self.perception(imgs, speeds)
        a_loss = -self.critic(feature, self.actor(feature)).mean()
        a_loss.backward()
        self.actor_optim.step()

        # train perception
        self.perc_optim.zero_grad()
        #p_loss = a_loss + c_loss
        #p_loss.backward()
        self.perc_optim.step()

        # update target
        self.update_target()

    def update_target(self):
        target = [self.actor_target, self.critic_target]
        source = [self.actor, self.critic]

        for t, s in zip(target, source):
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def run(self, img, speed, meter, train_state):
        if train_state > 0:
            img = np.expand_dims(img, axis=0)
            img = np.transpose(img, (0, 3, 1, 2))
            reshaped_speed = np.reshape(speed,(1, 1))
            img_tensor = tr.from_numpy(img).float()
            speed_tensor = tr.from_numpy(reshaped_speed).float()
            feature = self.perception(img_tensor, speed_tensor).detach()
            actions = self.actor(feature).detach().cpu().numpy()
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


class Perception(nn.Module):
    def __init__(self):
        super(Perception,self).__init__()
        self.img_extractor = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
        )
        self.speed_extractor = nn.Sequential(
            nn.Linear(1, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
    def forward(self, img, speed):
        img_perception = self.img_extractor(img)
        img_perception = img_perception.view(img_perception.size(0), -1)
        spd_perception = self.speed_extractor(speed)
        x = tr.cat([img_perception, spd_perception], dim=-1)
        return x


class Actor(nn.Module):
    def __init__(self, num_action):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(1216, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_action),
        )
    def forward(self, x):
        o = self.actor(x)
        return o


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.action_extractor = nn.Sequential(
            nn.Linear(2, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
        self.critic = nn.Sequential(
            nn.Linear(1248, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
    def forward(self, x, a):
        a = self.action_extractor(a)
        o = tr.cat([x, a], dim=-1)
        o = self.critic(o)
        return o

