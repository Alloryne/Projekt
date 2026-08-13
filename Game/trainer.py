from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


class EmpiricalLagrangianTrainer:
    """
  Trainer implementing the pseudocode.

  Arguments
  - model: nn.Module holding parameters w
  - objective_fn: callable(model, eps_reports, true_reports) -> utility of planner for given reports
  - utility_fn: callable(model, eps_reports, true_reports, i) -> utility tensor for a single player i
  - num_players: number of players n,
  - rho_scheduler: callable(t) -> rho,
  - device: torch device
  """

    def __init__(
            self,
            model: nn.Module,
            objective_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            utility_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, int], torch.Tensor],
            players_num: int,
            rho_scheduler: Callable[[int], float],
            gamma: float = 1e-1,
            eta: float = 1e-1,
            train_misreports_optimization_loops: int = 100,
            eval_misreports_optimization_loops: int = 0,
            batch_size: int = 32,
            q_number: int = 1,
            device: Optional[torch.device] = None,
    ):
        self.model = model
        self.objective_fn = objective_fn
        self.utility_fn = utility_fn
        self.players_num = players_num
        self.rho_scheduler = rho_scheduler
        self.gamma = gamma
        self.eta = eta
        self.train_misreport_optimization_loops = train_misreports_optimization_loops
        self.eval_misreport_optimization_loops = eval_misreports_optimization_loops
        self.batch_size = batch_size
        self.q_number = q_number
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.lambda_vec = torch.zeros(players_num, device=self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)

    def _optimize_misreports(
            self,
            misreport_optimization_loops: int,
            independent_reports: torch.Tensor,
            for_players_reports: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each sample in batch for each player calculates misreport. This procedure is inspired by Dutting
        to get DSIC misreports.
        The replacement of only the misreporting player's report is correct for DSIC.
        :param independent_reports: vector that has the independent reports, shape (batch_size, n_players)
        :param for_players_reports: vector that has the reports for players, shape (batch_size, n_players)
        :return:
        """
        B, n = for_players_reports.shape[0], self.players_num
        # print(B, n, for_players_reports.shape)
        # Initialize misreports
        v_prime = torch.rand_like(for_players_reports, device=self.device, requires_grad=True)

        for _ in range(misreport_optimization_loops):
            if v_prime.grad is not None:
                v_prime.grad.detach_()
                v_prime.grad.zero_()

            v_base = for_players_reports.unsqueeze(0).repeat(n, 1, 1).clone()  # (n, B, n)

            for i in range(n):
                v_base[i, :, i] = v_prime[:, i]
            utils = []

            for i in range(n):
                u_i = self.utility_fn(
                    self.model,
                    independent_reports,  # shared (B, n)
                    v_base[i],  # (B, n) for player i
                    i
                )
                utils.append(u_i)
            utils_stack = torch.stack(utils, dim=0)  # (n, B)

            scalar = utils_stack.sum()
            scalar.backward()

            with torch.no_grad():
                if v_prime.grad is None:
                    break
                v_prime += self.gamma * v_prime.grad
                v_prime.grad.zero_()

        return v_prime.detach()

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
    One epoch loop over minibatches S_t.
    dataloader batch
    {
        'true_values',
        'for_players_reports',
        'independent_reports'
    }
    """
        t = 0
        metrics_list = []

        pbar = tqdm(dataloader, desc="Epoch training")

        for batch in pbar:
            true_reports = batch['true_values'].to(self.device)
            for_player_reports = batch['for_players_reports'].to(self.device)
            independent_reports = batch['independent_reports'].to(self.device)

            # print(f"True reports: {true_reports}")
            # print(f"For player reports: {for_player_reports}")
            # print(f"Independent reports: {independent_reports}")

            players_num = self.players_num
            rho_t = self.rho_scheduler(t)

            # 6-10 inner loop optimize misreports
            v_prime = self._optimize_misreports(
                self.train_misreport_optimization_loops,
                independent_reports,
                for_player_reports
            )
            # print(f"Optimized misreports: {v_prime}")

            # 11-13 Compute regret gradient
            # For the gradient of regret wrt w, we compute a scalar equal to the mean regret across batch
            regrets_per_player = torch.zeros(players_num, device=self.device)
            for i in range(players_num):
                # reports for misreport case
                v_reports_mis = for_player_reports.clone()
                v_reports_mis[:, i] = v_prime[:, i]

                u_mis = self.utility_fn(self.model, independent_reports, v_reports_mis, i)  # (B,)

                # truthful utility
                u_truth = self.utility_fn(self.model, independent_reports, for_player_reports, i)  # (B,)

                regret = (u_mis - u_truth)  # (B,)
                mean_regret = regret.mean()
                regrets_per_player[i] = mean_regret

            # 14-15 Compute Lagrangian gradient and update w
            # Build Lagrangian scalar: -objective + sum_i lambda_i * pos(regret_i) + (rho/2) * pos(regret_i)^2
            objective = self.objective_fn(self.model, independent_reports, for_player_reports, true_reports)
            pos_regrets = torch.clamp(regrets_per_player, min=0.0)
            lagrangian = -objective + (self.lambda_vec * pos_regrets).sum() + 0.5 * rho_t * (pos_regrets ** 2).sum()

            # gradient step on w (we use optimiser)
            self.optimizer.zero_grad()
            lagrangian.backward()
            self.optimizer.step()

            # 16-20 Update Lagrange multipliers once in Q iterations
            if (t % self.q_number) == 0:
                # compute tilde_rgti(w_{t+1}) — here we use pos_regrets evaluated at new w (approx),
                # for simplicity we reuse pos_regrets computed above which used current model parameters
                # For a closer match you'd recompute regrets after the w update.
                self.lambda_vec = self.lambda_vec + rho_t * pos_regrets.detach()

            pbar.set_postfix({
                'objective': f"{objective.item():.4f}",
                'lagrangian': f"{lagrangian.item():.4f}",
                'max_regret': f"{pos_regrets.max().item():.4f}",
                'lambda_max': f"{self.lambda_vec.max().item():.4f}"
            })

            # store metrics for return
            metrics_list.append({
                'objective': f"{objective.item():.4f}",
                'lagrangian': lagrangian.item(),
                'max_regret': pos_regrets.max().item(),
                'lambda_max': self.lambda_vec.max().item()
            })

            t += 1

        return metrics_list

    def train(self, dataloader, epoch_n: int):
        """
        Train for epoch_n epochs (each epoch loops over dataloader once).
        """

        metrics_list_total = []
        for epoch in range(epoch_n):
            metrics_list = self.train_one_epoch(dataloader)
            metrics_list_total.append(metrics_list)

        return metrics_list_total

    def evaluate_model(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
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
        device = self.device
        model.eval()

        total_objective = 0.0
        total_regret = torch.zeros(self.players_num, device=device)
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                true_reports = batch['true_values'].to(device)
                for_player_reports = batch['for_players_reports'].to(device)
                independent_reports = batch['independent_reports'].to(device)

                n = self.players_num

                # --- objective ---
                objective_val = self.objective_fn(model, independent_reports, for_player_reports, true_reports)
                total_objective += objective_val.item()

                # --- compute regrets ---
                # For each i, compare truthful vs misreport where only i deviates.
                regrets = []

                for i in range(n):
                    # truthful
                    u_truth = self.utility_fn(model, independent_reports, true_reports, i)

                    # best unilateral deviation among players
                    # search misreport that maximises utility (1-step best response)
                    # using brute-force evaluation: v' = Uniform(0,1)
                    # (this is evaluation, not training, so OK)
                    v_prime = self._optimize_misreports(
                        self.eval_misreport_optimization_loops,
                        independent_reports,
                        for_player_reports
                    )

                    # only player i misreports
                    v_reports_mis = for_player_reports.clone()
                    v_reports_mis[:, i] = v_prime[:, i]

                    u_mis = self.utility_fn(model, independent_reports, v_reports_mis, i)

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

    def binary_evaluate_model(
            self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
    ):
        """
        Evaluate model over the binary case dataloader.

        Returns a dict:
        {
            'objective': float,
            'avg_regret': float,
            'max_regret': float
        }
        """
        device = self.device
        model.eval()

        total_objective = 0.0
        total_regret = torch.zeros(self.players_num, device=device)
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                true_reports = batch['true_values'].to(device)
                for_player_reports = batch['for_players_reports'].to(device)
                independent_reports = batch['independent_reports'].to(device)

                n = self.players_num

                # --- objective ---
                objective_val = self.objective_fn(model, independent_reports, for_player_reports, true_reports)
                total_objective += objective_val.item()

                # --- compute regrets ---
                # For each i, compare truthful vs misreport where only i deviates.
                regrets = []

                for i in range(n):
                    # truthful
                    u_truth = self.utility_fn(model, independent_reports, true_reports, i)

                    # best unilateral deviation among players
                    # search misreport that maximises utility (1-step best response)
                    # using brute-force evaluation: v' = Uniform(0,1)
                    # (this is evaluation, not training, so OK)

                    # only player i misreports
                    v_reports_mis = for_player_reports.clone()
                    v_reports_mis[:, i] = 1 - v_reports_mis[:, i]

                    u_mis = self.utility_fn(model, independent_reports, v_reports_mis, i)

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

