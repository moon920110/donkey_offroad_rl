import torch.nn as nn
import torch as tr
from torch.distributions import Categorical
import torch.nn.functional as F

class FixedCategorical(Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)
    def log_probs(self, actions):
        return (
                super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1))
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

