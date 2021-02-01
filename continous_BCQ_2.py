import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        self.speed_extractor = nn.Sequential(
            nn.Linear(1, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.acti = nn.Sequential(
            nn.Linear(1216 +2,64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, speed, action):
        state = self.features(state)
        state = state.reshape(state.shape[0], -1)
        s = self.speed_extractor(speed)

        a = self.acti(torch.cat([state, s, action], 1))
        a = self.phi * self.max_action * torch.tanh(a)
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        self.acti = nn.Sequential(
            nn.Linear(1216 +2,64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.acti2 = nn.Sequential(
            nn.Linear(1216 +2,64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.speed_extractor = nn.Sequential(
            nn.Linear(1, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, state, speed, action):
        state = self.features(state)
        state = state.reshape(state.shape[0], -1)
        s = self.speed_extractor(speed)

        q1 = self.acti(torch.cat([state, s, action], 1))

        q2 = self.acti2(torch.cat([state, s, action], 1))

        return q1, q2

    def q1(self, state, speed, action):
        state = self.features(state)
        state = state.reshape(state.shape[0], -1)
        s = self.speed_extractor(speed)

        q1 = self.acti(torch.cat([state, s, action], 1))
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        self.e1 = nn.Linear(1216 + action_dim, 750) #11264
        self.e2 = nn.Linear(750, 750)


        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(1216 + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

        self.speed_extractor = nn.Sequential(
            nn.Linear(1, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, state, speed, action):
        state2 = self.features(state)
        state2 = state2.reshape(state2.shape[0], -1)
        s = self.speed_extractor(speed)

        z = F.relu(self.e1(torch.cat([state2,s, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, speed, z)

        return u, mean, std

    def decode(self, state, speed, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        state = self.features(state)
        state = state.reshape(state.shape[0], -1)
        s = self.speed_extractor(speed)

        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state,s, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05): #0.005
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device

    def select_action(self, state, speed):

        with torch.no_grad():
            state = state.repeat(100, 1,1,1)
            speed = speed.repeat(100,1)
            action = self.actor(state, speed, self.vae.decode(state, speed ,z=None))
            q1 = self.critic.q1(state, speed, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done, speed, next_speed, _ = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, speed, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0)
                next_speed = torch.repeat_interleave(next_speed,10,0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, next_speed,
                                                          self.actor_target(next_state, next_speed, self.vae.decode(next_state, next_speed)))

                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                        target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, speed, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state,speed)
            perturbed_actions = self.actor(state, speed, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, speed, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename+"_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.vae.state_dict(), filename + "_vae")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.vae.load_state_dict(torch.load(filename + "_vae"))