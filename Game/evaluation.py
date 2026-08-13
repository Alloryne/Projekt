from typing import Callable

import torch
import torch.nn as nn


# Evaluation
class ArgmaxMechanism(nn.Module):
    """
    Mechanism that expects flattened input:
    shape (B, 2 * n_players)
    first n = independent_reports
    next  n = by_player_reports
    """

    def __init__(self, num_players):
        super().__init__()
        self.num_players = num_players

    def forward(self, reports):
        """
        reports: (B, 2*n_players)
        """
        B = reports.size(0)
        n = self.num_players
        independent_reports = reports[:, :n]  # (B, n)

        winner = independent_reports.argmax(dim=1)  # (B,)
        one_hot = torch.zeros_like(independent_reports)
        one_hot[torch.arange(B), winner] = 1.0
        return one_hot
