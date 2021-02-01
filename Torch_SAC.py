import torch.nn as nn
import torch as tr
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as trd
import random

from collections import deque
from baseagent import BaseAgent
steering_values=[-0.3, -0.15, 0, 0.15, 0.3],
throttle_values=[0, 0.25, 0.5, 0.75, 1]

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def save(self, img, speed, steer, throttle, rew, next_img, next_speed, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (img, speed, steer, throttle, rew, next_img, next_speed, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        img, speed, steer, throttle, rew, next_img, next_speed, mask = map(np.stack, zip(*batch))
        return img, speed, steer, throttle, rew, next_img, next_speed, mask

    def __len__(self):
        return len(self.buffer)

class SACAgent(BaseAgent):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None,
                 alpha=0.2, *args, **kwargs):
        super(SACAgent, self).__init__(*args, **kwargs)
        self.steer = [-0.3, -0.15, 0, 0.15, 0.3]
        self.throttle = [0, 0.25, 0.5, 0.75, 1]
        self.perception = Perception()
        self.actor= Actor(num_processed=1216, num_action=[len(self.steer), len(self.throttle)])
        self.actor_target = Actor(num_processed=1216, num_action=[len(self.steer), len(self.throttle)])
        self.critic = Critic(1216)
        self.critic_target = Critic(1216)
        # load model
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.tau = 1e-3
        self.evaluate = False
        # set optimizer
        self.lr = 0.001
        #self.decay = -5000
        self.actor_optim = optim.Adam(self.actor_target.parameters(), lr=self.lr,)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr, )
        self.perc_optim = optim.Adam(self.perception.parameters(), lr=self.lr,)

        # common settings
        self.gamma = 0.99
        self.memory = ReplayMemory(50000, 1)
        self.model_path = model_path
        self.n = 0
        self.train_step = 0
        self.r_sum = 0
        self.last_state = None
        self.last_actions = None
        self.batch_size = batch_size

        # about SAC
        self.alpha = alpha
        self.target_steer_entropy = -tr.prod(tr.Tensor(len(self.steer))).item()
        self.target_throttle_entropy = -tr.prod(tr.Tensor(len(self.throttle))).item()
        self.log_alpha = tr.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)

    def save(self):
        tr.save(self.actor.state_dict(),'{}_actor.h5'.format(self.model_path))
        tr.save(self.critic.state_dict(), '{}_critic.h5'.format(self.model_path))

    def load(self,model_paths):
        self.actor = tr.load(model_paths[0])
        self.actor_target = tr.load(model_paths[1])
        self.critic = tr.load(model_paths[2])
        self.critic_target = tr.load(model_paths[3])

    def _act(self, img, scalar):
        processed = self.perception(img, scalar)
        if self.evaluate is False:
            action, _ = self.actor(processed)
        else:
            _, action = self.actor(processed)
        action = [action[0].detach().cpu().numpy()[0], action[1].detach().cpu().numpy()[0]]
        return action

    def train(self):
        # sample in Memory
        batches = self.memory.sample(batch_size=self.batch_size)
        batches = np.array(batches).transpose()
        imgs = np.vstack(batches[0])
        speeds = np.vstack(batches[1])
        steers = np.vstack(batches[2])
        throttles = np.vstack(batches[3])
        rews = np.vstack(batches[4])
        imgs_next = np.vstack(batches[5])
        speeds_next = np.vstack(batches[6])
        speeds = np.reshape(speeds, (-1, 1))
        masks = np.vstack(batches[7])
        # convert to torch tensor
        imgs = tr.from_numpy(imgs).float()
        speeds = tr.from_numpy(speeds).float()
        rews = tr.from_numpy(rews).float()
        steers = tr.from_numpy(steers).float()
        throttles = tr.from_numpy(throttles).float()
        masks = tr.from_numpy(masks).float()

        # train critic
        with tr.no_grad():
            next_processed = self.perception(imgs_next,speeds_next)
            next_actions, next_logprobs = self.actor(next_processed)
            qs1_target, qs2_target = self.critic_target(next_processed, next_actions)
            q_target = tr.min(qs1_target, qs2_target) - self.alpha * next_logprobs
            next_q_value = rews + masks * self.gamma * (q_target)
        processed = self.perception(imgs, speeds)
        qs1,qs2 = self.critic(processed, actions)
        qs1_loss = F.mse_loss(qs1,next_q_value)
        qs2_loss = F.mse_loss(qs2, next_q_value)
        critic_loss = qs1_loss +qs2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # train actor
        processed = self.perception(imgs, speeds)
        actions, logprobs = self.actor(processed)
        q1, q2 = self.critic(processed, actions)
        q = tr.min(q1,q2)
        actor_loss = ((self.alpha * logprobs) - q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()
        return qs1_loss.item(), qs2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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
            actions = self._act(img_tensor, speed_tensor)
            a_t = [actions[0][0], actions[1][0]] # steering, throttle

            if train_state < 4:
                if self.train_step > 0:
                    self.n += 1

                    reward = meter - self.train_step * 0.01
                    if train_state == 2:
                        reward -= 100
                    elif train_state == 3:
                        reward += 100
                    done = [train_state > 1]
                    self.r_sum += reward
                    mask = tr.FloatTensor([0.0 if done_ else 1.0 for done_ in done])
                    self.memory.save(self.last_state['img'], self.last_state['speed'], self.last_actions[0], self.last_actions[1], reward, img, speed, mask)

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
    def __init__(self, num_processed, num_action, num_hidden=128):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_processed,num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden,num_hidden),
            nn.Tanh()
        )
        self.steer_mu = nn.Sequential(nn.Linear(num_hidden,num_action[0]),nn.Tanh())
        self.steer_std = nn.Sequential(nn.Linear(num_hidden,num_action[0]),nn.Tanh())

        self.throttle_mu = nn.Sequential(nn.Linear(num_hidden,num_action[1]),nn.Tanh())
        self.throttle_std = nn.Sequential(nn.Linear(num_hidden, num_action[1]), nn.Tanh())
        self.tanh = nn.Tanh()
    def forward(self, processed):
        hidden_actor = self.actor(processed)
        str_mu = self.steer_mu(hidden_actor)
        str_std = self.steer_std(hidden_actor)
        thr_mu = self.throttle_mu(hidden_actor)
        thr_std = self.throttle_std(hidden_actor)
        str_logstd = tr.clamp(str_std,-20,2)
        thr_logstd = tr.clamp(thr_std,-20,2)
        str_std = tr.exp(str_logstd)
        thr_std = tr.exp(thr_logstd)
        str_dist = trd.Normal(str_mu,str_std)
        thr_dist = trd.Normal(thr_mu,thr_std)
        steers = self.tanh(str_dist.sample())
        throttles = self.tanh(thr_dist.sample())
        str_logprobs = str_dist.log_prob(steers)
        thr_logprobs = thr_dist.log_prob(throttles)
        actions = [steers,throttles]
        logprobs = [str_logprobs, thr_logprobs]
        return actions,logprobs

class Critic(nn.Module):
    def __init__(self, num_processed, num_hidden=128):
        super(Critic,self).__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(num_processed, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(num_processed, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, processed, actions):
        combined = tr.cat([processed, actions], 1)
        qs1 = self.critic1(combined)
        qs2 = self.critic(combined)
        return qs1,qs2
