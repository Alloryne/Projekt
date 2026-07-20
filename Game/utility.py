import torch
import torch.nn as nn


def utility_fn(
        _model: nn.Module,
        independent_reports: torch.Tensor,
        self_reports: torch.Tensor,
        i: int
) -> torch.Tensor:
    """
    Calculates the utility achieved by the ith player under the given model, independent reports and self reports.
    :param _model: Model of the mechanism
    :param independent_reports: vector that has the independent reports
    :param self_reports: vector that has the self reports
    :param i: the index of the player
    :return: the probability of choosing the given player
    """
    B = independent_reports.shape[0]
    v = torch.cat([independent_reports, self_reports], dim=-1).view(B, -1)  # shape (B, n_players*2)
    outcome = _model(v)  # (B, n_players)
    return outcome[:, i]


# Utility function for the planner (value of returned report)
def objective_fn(
        _model: nn.Module,
        independent_reports: torch.Tensor,
        true_reports: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the utility achieved by the planner under the given model, independent reports and true reports.
    :param _model: Model of the mechanism
    :param independent_reports: vector that has the independent reports
    :param true_reports: vector that has the true reports
    :return: Utility achieved by the planner.
    """
    B = independent_reports.shape[0]
    v = torch.cat([independent_reports, true_reports], dim=-1).view(B, -1)
    outcome = _model(v)  # (B, n_players)
    planner_util = (outcome * true_reports).sum(dim=1)
    return planner_util.mean()
