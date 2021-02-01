import torch.nn as nn
import torch as tr
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from distributions import Categorical
import random

from collections import deque
from baseagent import BaseAgent
steering_values=[-0.3, -0.15, 0, 0.15, 0.3],
throttle_values=[0, 0.25, 0.5, 0.75, 1]

class Memory:
    def __init__(self, batch_size, img_shape):
        self.batch_size = batch_size
        self.imgs = tr.zeros(batch_size+1, 3, 120, 160)
        self.speeds = tr.zeros(batch_size+1,1)
        self.throttles = tr.zeros(batch_size,1)
        self.steers = tr.zeros(batch_size,1)
        self.rews = tr.zeros(batch_size,1)
        self.vals = tr.zeros(batch_size+1,1)
        self.rets = tr.zeros(batch_size+1,1)
        self.steer_logprobs = tr.zeros(batch_size,1)
        self.throttle_logprobs = tr.zeros(batch_size,1)
        self.masks = tr.ones(batch_size+1, 1)
        self.bad_masks = tr.ones(batch_size+1, 1)
        self.step = 0

    def save(self, img, speed, acts, rew, logprobs, v, mask, bad_mask):
        self.imgs[self.step + 1].copy_(img)
        self.speeds[self.step + 1].copy_(speed)
        self.throttles[self.step].copy_(acts[1])
        self.steers[self.step].copy_(acts[0])
        self.rews[self.step].copy_(rew)
        self.vals[self.step].copy_(v)
        self.steer_logprobs[self.step].copy_(logprobs[0])
        self.throttle_logprobs[self.step].copy_(logprobs[1])
        self.masks[self.step + 1].copy_(mask)
        self.bad_masks[self.step + 1].copy_(bad_mask)
        self.step = (self.step + 1) % self.batch_size

    def after_train(self):
        self.imgs[0].copy_(self.imgs[-1])
        self.speeds[0].copy_(self.speeds[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_val, use_gae, gamma, gae_lambda):
        if use_gae:
            self.vals[-1] = next_val
            gae = 0
            for step in reversed(range(self.rews.size(0))):
                delta = self.rews[step] + gamma * self.vals[step + 1] * self.masks[step + 1] - self.vals[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                gae = gae * self.bad_masks[step + 1]
                self.rets[step] = gae + self.vals[step]
        else:
            self.rets[-1] = next_val
            for step in reversed(range(self.rews.size(0))):
                self.rets[step] = (self.rets[step + 1] * gamma * self.masks[step + 1] + self.rets[step]) * \
                                  self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.vals[step]
    def clear(self):
        self.step = 0

    def sample(self):
        batch = []
        batch.append(self.imgs[:self.step])
        batch.append(self.speeds[:self.step])
        batch.append(tr.cat([self.steers[:self.step],self.throttles[:self.step]],axis=1))
        batch.append(self.rets[:self.step + 1])
        batch.append(tr.cat([self.steer_logprobs[:self.step],self.throttle_logprobs[:self.step]],axis=-1))
        batch.append(self.vals[:self.step + 1])
        return batch

class PPOAgent(BaseAgent):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, k=4,
                 clip=0.2, use_clipped=True, entropy_coef=0.01, max_grad_norm=0.5, value_loss_coef=0.5, *args, **kwargs):
        super(PPOAgent, self).__init__(*args, **kwargs)
        self.steer = [-0.3, -0.15, 0, 0.15, 0.3]
        self.throttle = [0, 0.25, 0.5, 0.75, 1]
        self.perception = Perception()
        self.actor_critic = ActorCritic(num_processed=1216,num_hidden=128)
        self.actor_critic_target = ActorCritic(num_processed=1216, num_hidden=128)
        # load model
        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
        self.tau = 1e-3

        # set optimizer
        self.lr = 0.001
        #self.decay = -5000
        self.actor_critic_optim = optim.Adam(self.actor_critic.parameters(), lr=self.lr,)
        self.perc_optim = optim.Adam(self.perception.parameters(), lr=self.lr,)

        # common settings
        self.gamma = 0.99
        self.memory = Memory(batch_size=batch_size, img_shape=input_shape)
        self.model_path = model_path
        self.n = 0
        self.train_step = 0
        self.r_sum = 0
        self.last_state = None
        self.last_actions = None
        self.batch_size = batch_size

        # about PPO
        self.k = k
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.use_clipped = use_clipped
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.dist1 = Categorical(self.actor_critic.num_hidden, len(self.steer))
        self.dist2 = Categorical(self.actor_critic.num_hidden, len(self.throttle))
    def save(self):
        tr.save(self.actor_critic.state_dict(),'{}_actorcritic.h5'.format(self.model_path))

    def load(self,model_paths):
        self.actor_critic = tr.load(model_paths[0])
        self.actor_critic_target = tr.load(model_paths[1])
    def _evaluate(self, imgs, scalars, actions):
        processed = self.perception(imgs, scalars)
        vs, hidden_actor = self.actor_critic(processed)
        steer_dist = self.dist1(hidden_actor)
        throttle_dist = self.dist2(hidden_actor)

        # actions = [steer, throttle]
        throttles = actions[:, 1]
        steers = actions[:, 0]
        steer_logprobs = steer_dist.log_probs(steers)
        throttle_logprobs = throttle_dist.log_probs(throttles)
        steer_dist_entropy = steer_dist.entropy()
        throttle_dist_entropy = throttle_dist.entropy()
        logprobs = [steer_logprobs, throttle_logprobs]
        dist_entropy = tr.cat([steer_dist_entropy, throttle_dist_entropy])
        dist_entropy = dist_entropy.mean(axis=-1)
        return vs, logprobs, dist_entropy

    def _act(self, img, scalar, deterministic=False):
        processed = self.perception(img, scalar)
        v, hidden_actor = self.actor_critic(processed)
        steer_dist = self.dist1(hidden_actor)
        throttle_dist = self.dist2(hidden_actor)
        if deterministic:
            steer = steer_dist.mode()
            throttle = throttle_dist.mode()
        else:
            steer = steer_dist.sample()
            throttle = throttle_dist.sample()
        steer_logprobs = steer_dist.log_probs(steer)
        throttle_logprobs = throttle_dist.log_probs(throttle)
        action = [steer.detach().cpu().numpy(), throttle.detach().cpu().numpy()]
        logprobs = [steer_logprobs.detach(), throttle_logprobs.detach()]
        return v.detach(), action, logprobs

    def train(self):
        # sample in Memory
        batches = self.memory.sample()
        imgs = batches[0]
        speeds = batches[1]
        actions = batches[2]
        returns = batches[3]
        old_act_logprobs = batches[4]
        vs_preds = batches[5]


        # convert to torch tensor

        advs = returns[:-1] - vs_preds[:-1]
        advs = (advs - advs.mean()) / (advs.std()+1e-5)
        advs = advs.view(-1, 1)
        value_loss_total = 0
        actor_loss_total = 0
        dist_entorpy_total = 0
        vs_preds = vs_preds[:-1]
        returns = returns[:-1]
        # train for k times
        for k in range(self.k):
            vs, act_logprobs, dist = self._evaluate(imgs, speeds, actions)
            act_logprobs = tr.cat(act_logprobs, axis=-1)
            ratio = tr.exp(act_logprobs - old_act_logprobs)
            surr1 = ratio * advs
            surr2 = tr.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
            actor_loss = -tr.min(surr1, surr2).mean()
            if self.use_clipped:
                value_pred_clipped = vs_preds + (vs - vs_preds).clamp(-self.clip,self.clip)
                value_loss = (vs - returns).pow(2)
                value_loss_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * tr.max(value_loss,value_loss_clipped).mean()
            else:
                value_loss = 0.5 * (returns -vs).pow(2).mean()
            self.actor_critic_optim.zero_grad()
 
            (value_loss * self.value_loss_coef + actor_loss -
             dist * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.actor_critic_optim.step()

            # train perception
            self.perc_optim.zero_grad()
            self.perc_optim.step()

            value_loss_total += value_loss.item()
            actor_loss_total += actor_loss.item()
            dist_entorpy_total += dist.item()
        # update target
        self.update_target()
        num_updates = self.k * self.batch_size
        value_loss_total /= num_updates
        actor_loss_total /= num_updates
        dist_entorpy_total /= num_updates
        self.memory.clear()
        return value_loss_total, actor_loss_total, dist_entorpy_total

    def update_target(self):
        target = [self.actor_critic_target]
        source = [self.actor_critic]

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
            v, actions, logprobs = self._act(img_tensor, speed_tensor)
            a_t = [self.steer[int(actions[0][0])], self.throttle[int(actions[1][0])]]
            print(a_t)
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
                    bad_mask = tr.FloatTensor([1.0])
                    self.memory.save(tr.tensor(self.last_state['img'][0]), tr.tensor(self.last_state['speed']), \
                                    tr.tensor(self.last_actions), tr.tensor(reward), tr.tensor(logprobs), \
                                    tr.tensor([v]), mask, bad_mask)
                    if self.train_step % self.batch_size == self.batch_size - 1:
                        self.train_step += 1
                        self.last_state = {
                                'img': img,
                            'speed': speed,
                            }
                        self.last_actions = a_t
                        return 0, 0, True
                    if done[0]:
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
                if self.train_step >= self.batch_size:
                    print("TRAIN START!")
                    self.train()
                    print("TRAIN DONE!")
                    self.save()
                    print("SAVE DONE!")
                    self.train_step = 0
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


class ActorCritic(nn.Module):
    def __init__(self, num_processed, num_hidden=128):
        super(ActorCritic,self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_processed,num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden,num_hidden),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(num_processed, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh()
        )
        self.num_hidden = num_hidden
        self.critic_linear = nn.Linear(num_hidden,1)
    def forward(self, processed):
        hidden_critic = self.critic(processed)
        hidden_actor = self.actor(processed)
        return self.critic_linear(hidden_critic), hidden_actor
