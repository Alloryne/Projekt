import torch
import torch.nn as nn


def utility_fn(
        _model: nn.Module,
        independent_reports: torch.Tensor,
        by_player_reports: torch.Tensor,
        i: int
) -> torch.Tensor:
    """
    Calculates the utility achieved by the ith player under the given model, independent reports and self reports.
    :param _model: Model of the mechanism
    :param independent_reports: vector that has the independent reports
    :param by_player_reports: vector that has the values reported by the players
    :param i: the index of the player
    :return: the probability of choosing the given player
    """
    B = independent_reports.shape[0]
    v = torch.cat([independent_reports, by_player_reports], dim=-1).view(B, -1)  # shape (B, n_players*2)
    outcome = _model(v)  # (B, n_players)
    return outcome[:, i]


# Utility function for the planner (value of returned report)
def objective_fn(
        _model: nn.Module,
        independent_reports: torch.Tensor,
        by_player_reports: torch.Tensor,
        true_reports: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the utility achieved by the planner under the given model, independent reports and true reports.
    :param _model: Model of the mechanism
    :param independent_reports: vector that has the independent reports
    :param by_player_reports: vector that has the values reported by the players
    :param true_reports: vector that has the true reports
    :return: Utility achieved by the planner.
    """
    B = independent_reports.shape[0]
    v = torch.cat([independent_reports, by_player_reports], dim=-1).view(B, -1)
    outcome = _model(v)  # (B, n_players)
    planner_util = (outcome * true_reports).sum(dim=1)
    return planner_util.mean()
