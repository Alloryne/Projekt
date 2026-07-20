import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A general model that represents a parametrised mechanism for allocating the reward to player.
    It takes input as a vector that has first values reported for the players and then values from independent report.
    Returns probabilites of choosing a given player.
    """
    def __init__(self, n_players, num_hidden_layers=1):
        super().__init__()
        self.n_players = n_players

        layers = []
        input_dim = n_players * 2

        layers.append(nn.Linear(input_dim, n_players))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(n_players, n_players))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(n_players, n_players))

        self.net = nn.Sequential(*layers)

    def forward(self, reports):
        x = self.net(reports)
        return torch.softmax(x, dim=1)
