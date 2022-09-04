import torch
from torch import nn, Tensor
import math

class TimeEncoder(nn.Module):
    """
        Learned Position Encoder. Takes tensor of positional indicies and converts to learned embeddings
    """

    def __init__(self, n_timesteps, d_model):
        super().__init__()
        self.embeddor = nn.Embedding(n_timesteps, d_model) # lookup table, each with vector of size d_model
        nn.init.uniform_(self.embeddor.weight)

    def forward(self, pos_indicies):
        pos_indicies = pos_indicies.long()
        return self.embeddor(pos_indicies)

class SensorEncoder(nn.Module):
    """
        Learned Position Encoder. Takes tensor of positional indicies and converts to learned embeddings
    """

    def __init__(self, n_sensors, d_model):
        super().__init__()
        self.embeddor = nn.Embedding(n_sensors, d_model) # lookup table, each with vector of size d_model
        nn.init.uniform_(self.embeddor.weight)

    def forward(self, pos_indicies):
        pos_indicies = pos_indicies.long()
        return self.embeddor(pos_indicies)

class TrajEncoder(nn.Module):
    """
        Learned Position Encoder. Takes tensor of positional indicies and converts to learned embeddings
    """

    def __init__(self, n_traj, d_model):
        super().__init__()
        self.embeddor = nn.Embedding(n_traj, d_model) # lookup table, each with vector of size d_model
        nn.init.uniform_(self.embeddor.weight)

    def forward(self, pos_indicies):
        pos_indicies = pos_indicies.long()
        return self.embeddor(pos_indicies)
