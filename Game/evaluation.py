import torch
import torch.nn as nn


# Evaluation
class ArgmaxMechanism(nn.Module):
    """
    Mechanism that expects flattened input:
    shape (B, 2 * n_players)
    first n = eps
    next  n = reports
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
        eps = reports[:, :n]  # (B, n)
        vals = reports[:, n:]  # (B, n)

        winner = eps.argmax(dim=1)  # (B,)
        one_hot = torch.zeros_like(eps)
        one_hot[torch.arange(B), winner] = 1.0
        return one_hot


def evaluate_model(
        model: nn.Module,
        dataloader,
        objective_fn,
        utility_fn,
        num_players: int,
        device: torch.device
):
    """
    Evaluate model over the dataloader.

    Returns a dict:
    {
        'objective': float,
        'avg_regret': float,
        'max_regret': float
    }
    """
    model.eval()
    total_objective = 0.0
    total_regret = torch.zeros(num_players, device=device)
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            eps = batch["errors"].to(device)
            true_reports = batch["true_values"].to(device)

            B = true_reports.shape[0]
            n = num_players

            # noisy reports
            eps_reports = eps + true_reports

            # --- objective ---
            objective_val = objective_fn(model, eps_reports, true_reports)
            total_objective += objective_val.item()

            # --- compute regrets ---
            # For each i, compare truthful vs misreport where only i deviates.
            regrets = []

            for i in range(n):
                # truthful
                u_truth = utility_fn(model, eps_reports, true_reports, i)

                # best unilateral deviation among players
                # search misreport that maximizes utility (1-step best response)
                # using brute-force evaluation: v' = Uniform(0,1)
                # (this is evaluation, not training, so OK)
                v_prime = torch.rand_like(true_reports)

                # only player i misreports
                v_reports_mis = true_reports.clone()
                v_reports_mis[:, i] = v_prime[:, i]

                u_mis = utility_fn(model, eps_reports, v_reports_mis, i)

                regret_i = (u_mis - u_truth).mean()
                regrets.append(regret_i)

            regrets = torch.stack(regrets)  # (n,)
            total_regret += regrets
            total_batches += 1

    # aggregate
    avg_objective = total_objective / total_batches
    avg_regret_per_player = (total_regret / total_batches).cpu().numpy()
    max_regret = avg_regret_per_player.max()

    return {
        "objective": avg_objective,
        "avg_regret": avg_regret_per_player.mean(),
        "max_regret": max_regret,
        "regret_per_player": avg_regret_per_player
    }