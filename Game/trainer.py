from enum import Enum
from typing import Callable, Optional
import abc
from abc import abstractmethod

import torch
import torch.nn as nn
from tqdm import tqdm


class ReportStrategy(Enum):
    SELF_REPORT = 1
    OTHER_REPORT = 2


class EmpiricalLagrangianTrainer(abc.ABC):
    """
    Trainer for training a planner objective-maximising mechanism
    for selecting a winner.

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
            q_number: int = 1,
            device: Optional[torch.device] = None,
            report_strategy: ReportStrategy = ReportStrategy.SELF_REPORT
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
        self.q_number = q_number
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.report_strategy = report_strategy

        self.model.to(self.device)

        self.lambda_vec = torch.zeros(players_num, device=self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)

    @abstractmethod
    def _set_sample(
            self,
            batch
    ):
        self.true_reports = batch['true_values'].to(self.device)
        self.for_player_reports = batch['for_player_reports'].to(self.device)
        self.independent_reports = batch['independent_reports'].to(self.device)
        self.next_player_reports = batch.get('for_next_player_reports')
        self.next_player_reports = self.next_player_reports.to(self.device) if self.next_player_reports is not None else None

        self.B = self.for_player_reports.shape[0]


    @abstractmethod
    def _optimize_misreports(
            self,
            misreport_optimization_loops: int,
    ) -> torch.Tensor:
        misreports_prime = torch.rand_like(self.for_player_reports, device=self.device, requires_grad=True)
        return misreports_prime.detach()

    @abstractmethod
    def _get_regrets_from_misreports(
            self,
            misreports_prime
    ) -> torch.Tensor:
        return torch.zeros(self.players_num, device=self.device)

    @abstractmethod
    def _get_objective(
            self
    ) -> torch.Tensor:
        if self.next_player_reports is None:
            honest_reports = self.for_player_reports.clone()
        else:
            honest_reports = self.next_player_reports.clone()
        return self.objective_fn(self.model, self.independent_reports, honest_reports, self.true_reports)
    
    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
        One epoch loop over minibatches S_t.
        dataloader batch
        {
            'true_values',
            'for_player_reports',
            'for_next_player_reports',
            'independent_reports'
        }
        """
        t = 0
        metrics_list = []

        pbar = tqdm(dataloader, desc="Epoch training")

        for batch in pbar:
            self._set_sample(
                batch
            )

            # print(f"True reports: {true_reports}")
            # print(f"For player reports: {for_player_reports}")
            # print(f"Independent reports: {independent_reports}")
            # print(f"For next player reports: {next_player_reports}")
            rho_t = self.rho_scheduler(t)

            # 6-10 inner loop optimize misreports
            misreports_prime = self._optimize_misreports(
                self.train_misreport_optimization_loops,
            )
            # print(f"Optimized misreports: {misreports_prime}")

            # 11-13 Compute regret gradient
            # For the gradient of regret wrt w, we compute a scalar equal to the mean regret across batch
            regrets_per_player = self._get_regrets_from_misreports(self.for_player_reports)

            # 14-15 Compute Lagrangian gradient and update w
            # Build Lagrangian scalar: -objective + sum_i lambda_i * pos(regret_i) + (rho/2) * pos(regret_i)^2
            objective = self._get_objective()
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
                self._set_sample(batch)

                # --- objective ---
                objective = self._get_objective()
                total_objective += objective.item()

                # --- compute regrets ---
                # For each i, compare truthful vs misreport where only i deviates.
                regrets = []

                misreports_prime = self._optimize_misreports(
                    self.eval_misreport_optimization_loops,

                )

                regrets_per_player = self._get_regrets_from_misreports()
                total_regret += regrets_per_player
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

    @abstractmethod
    def _binary_get_regrets(self):
        return torch.zeros(self.players_num, device=self.device)

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
                self._set_sample(batch)

                # --- objective ---
                objective = self._get_objective()
                total_objective += objective.item()

                # --- compute regrets ---
                regrets = self._binary_get_regrets()  # (n,)
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


class DSICEmpiricalLagrangianTrainer(EmpiricalLagrangianTrainer):
    """
    Trainer for training a DSIC approximating planner objective-maximising mechanism
    for selecting a winner.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def _set_sample(
            self,
            batch
    ):
        super()._set_sample(batch)

    def _optimize_misreports(
            self,
            misreport_optimization_loops: int,
    ) -> torch.Tensor:
        """
        For each sample in batch for each player calculates misreport maximising regret.
        If trainer has DSIC constraint type then a player misreport is a vector that maximizes ex post regret:
        expected loss of utility when reporting truthfully, compared to when reporting optimally with knowledge
        of other players re ports..
        :param misreport_optimization_loops:
        self.independent_reports: vector that has the independent reports, shape (B, n)
        self.for_player_reports: vector that has the reports for players, shape (B, n)
        self.next_player_reports: vector that has the next reports, shape (B, n)
        :return: a tensor of size (B, n)
        """
        B, n = self.B, self.players_num

        if self.next_player_reports is None:
            honest_reports = self.for_player_reports.clone()
        else:
            honest_reports = self.next_player_reports.clone()

        # Initialize misreports
        misreports_prime = torch.rand_like(honest_reports, device=self.device, requires_grad=True)

        for _ in range(misreport_optimization_loops):
            if misreports_prime.grad is not None:
                misreports_prime.grad.detach_()
                misreports_prime.grad.zero_()

            reports_with_misreports = honest_reports.unsqueeze(0).repeat(n, 1, 1).clone()  # (n, B, n)

            for i in range(n):
                reports_with_misreports[i, :, i] = misreports_prime[:, i]
            utils = []

            for i in range(n):
                u_i = self.utility_fn(
                    self.model,
                    self.independent_reports,  # shared (B, n)
                    reports_with_misreports[i],  # (B, n) for player i
                    i
                )
                utils.append(u_i)
            utils_stack = torch.stack(utils, dim=0)  # (n, B)

            scalar = utils_stack.sum()
            scalar.backward()

            with torch.no_grad():
                if misreports_prime.grad is None:
                    break
                misreports_prime += self.gamma * misreports_prime.grad
                misreports_prime.grad.zero_()

        return misreports_prime.detach()

    def _get_regrets_from_misreports(
            self,
            misreports_prime
    ) -> torch.Tensor:
        n = self.players_num

        regrets_per_player = torch.zeros(n, device=self.device)
        
        if self.next_player_reports is None:
            honest_reports = self.for_player_reports.clone()
        else:
            honest_reports = self.next_player_reports.clone()
            
        for i in range(n):
            # reports for misreport case
            v_reports_mis = honest_reports.clone()
            v_reports_mis[:, i] = misreports_prime[:, i]

            u_mis = self.utility_fn(self.model, self.independent_reports, v_reports_mis, i)  # (B,)

            # truthful utility
            u_truth = self.utility_fn(self.model, self.independent_reports, honest_reports, i)  # (B,)

            regret = (u_mis - u_truth)  # (B,)
            mean_regret = regret.mean()
            regrets_per_player[i] = mean_regret
        return regrets_per_player

    def _get_objective(
            self,
    ) -> torch.Tensor:
        if self.next_player_reports is None:
            honest_reports = self.for_player_reports.clone()
        else:
            honest_reports = self.next_player_reports.clone()
        return self.objective_fn(self.model, self.independent_reports, honest_reports, self.true_reports)

    def _binary_get_regrets(
            self,
    ) -> torch.Tensor:
        n = self.players_num
        model = self.model

        regrets = []

        if self.next_player_reports is None:
            honest_reports = self.for_player_reports.clone()
        else:
            honest_reports = self.next_player_reports.clone()

        for i in range(n):
            u_truth = self.utility_fn(model, self.independent_reports, honest_reports, i)

            v_reports_0 = honest_reports.clone()
            v_reports_1 = honest_reports.clone()

            v_reports_0[:, i] = 0
            v_reports_1[:, i] = 1

            u_0 = self.utility_fn(model, self.independent_reports, v_reports_0, i)
            u_1 = self.utility_fn(model, self.independent_reports, v_reports_1, i)
            u_mis = max(u_0, u_1)

            regret_i = (u_mis - u_truth).mean()
            regrets.append(regret_i)
        return torch.stack(regrets)


class BICEmpiricalLagrangianTrainer(EmpiricalLagrangianTrainer):
    """
    Trainer for training a BIC approximating planner objective-maximising mechanism
    for selecting a winner.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def _set_sample(
            self,
            batch
    ):
        super()._set_sample(batch)
        # Prepare sample properly for other functions
        # Shape of reports: (B, n), first dimension is the batch dimension,
        # second is the player receiving report.
        # Shape of creted samples is (n, B, B, n)
        # First dimension is the player for whom the sample is used to calculate regret
        # The second dimension is from which batched sample we take the value of reports received by them
        # The third is batch dimension, fourth the player dimension of the tensor sent to mechanism.

        n, B = self.players_num, self.B
        self.true_base = (
            self.true_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.independent_base = (
            self.independent_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.for_player_base = (
            self.for_player_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.next_player_base = (
            self.next_player_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )

        if self.report_strategy == ReportStrategy.SELF_REPORT:
            for i in range(n):
                for j in range(B):
                    self.for_player_base[i, j, :, i] = self.for_player_reports[i, j]
                    self.independent_base[i, j, :, i] = self.independent_reports[i, j]
                    self.true_base[i, j, :, i] = self.true_reports[i, j]
        if self.report_strategy == ReportStrategy.OTHER_REPORT:
            for i in range(n):
                for j in range(B):
                    self.for_player_base[i, j, :, i] = self.for_player_reports[i, j]
                    self.independent_base[i, j, :, i] = self.independent_reports[i, j]
                    self.true_base[i, j, :, i] = self.true_reports[i, j]
                    self.next_player_base[i, j, :, i] = self.next_player_reports[i, j]

                    # Set next player
                    self.true_base[i, j, :, i + 1 % n] = self.true_reports[i + 1 % n, j]
                    self.for_player_base[i, j, :, i + 1 % n] = self.for_player_reports[i + 1 % n, j]
                    self.independent_base[i, j, :, i + 1 % n] = self.independent_reports[i + 1 % n, j]

                    # Set previous player
                    self.next_player_base[i, j, :, i - 1 % n] = self.next_player_reports[i - 1 % n, j]

    def _optimize_misreports(
            self,
            misreport_optimization_loops: int
    ) -> torch.Tensor:
        """
        For each sample in batch for each player calculates misreports maximising regret.
        A player misreport is a vector that maximizes interim regret:
        expected loss of utility when reporting truthfully, compared to when reporting optimally without knowledge
        of other players' reports (measured as expectation).
        :param misreport_optimization_loops:
        self.independent_reports: vector that has the independent reports, shape (B, n)
        self.for_player_reports: vector that has the reports for players, shape (B, n)
        :return: a tensor of size (B, n), for each player and batched game
        this is a misreport that maximises expected utility over all games where the player gets the value
        from the batched game.
        """
        B, n = self.for_player_reports.shape[0], self.players_num

        # Initialize misreports
        misreports_prime = torch.rand_like(self.for_player_reports, device=self.device, requires_grad=True)

        for _ in range(misreport_optimization_loops):
            if misreports_prime.grad is not None:
                misreports_prime.grad.detach_()
                misreports_prime.grad.zero_()

            if self.next_player_reports is None:
                reports_with_misreports = self.for_player_base.clone()
            else:
                reports_with_misreports = self.next_player_base.clone()
            # (n, B, B, n)

            # TODO Replace loops with vectorized operations
            for i in range(n):
                for j in range(B):
                    reports_with_misreports[i, j, :, i] = misreports_prime[j, i]

            utils = []

            for i in range(n):
                for j in range(B):
                    u_i_j = self.utility_fn(
                        self.model,
                        self.independent_base[i, j],  # shared (B, n)
                        reports_with_misreports[i, j],  # (B, n) for player i
                        i
                    )
                    utils.append(u_i_j)
            utils_stack = torch.stack(utils, dim=0)  # (n*B, B)

            scalar = utils_stack.sum()
            scalar.backward()

            with torch.no_grad():
                if misreports_prime.grad is None:
                    break
                misreports_prime += self.gamma * misreports_prime.grad
                misreports_prime.grad.zero_()

        return misreports_prime.detach()

    def _get_regrets_from_misreports(
            self,
            misreports_prime,
    ) -> torch.Tensor:
        B, n = self.for_player_reports.shape[0], self.players_num

        regrets_per_player = torch.zeros(n, device=self.device)

        if self.next_player_reports is None:
            honest_base = self.for_player_base.clone()
        else:
            honest_base = self.next_player_base.clone()

        misreport_base = honest_base.clone()

        for i in range(n):
            for j in range(B):
                misreport_base[i, j, :, i] = misreports_prime[i, j]

        for i in range(n):
            regrets = []
            for j in range(B):
                # reports for misreport case
                u_mis = self.utility_fn(self.model, self.independent_base[i, j], misreport_base[i, j], i)  # (B,)

                # truthful utility
                u_truth = self.utility_fn(self.model, self.independent_base[i, j], honest_base[i, j], i)  # (B,)

                regret = (u_mis - u_truth)  # (B,)
                regrets.append(regret)
            regrets = torch.stack(regrets, dim=0)
            mean_regret = regrets.mean()
            regrets_per_player[i] = mean_regret
        return regrets_per_player

    def _get_objective(
            self,
    ) -> torch.Tensor:
        objectives = []
        for i in range(self.players_num):
            for j in range(self.for_player_reports.shape[0]):
                objective_i_j = self.objective_fn(
                    self.model,
                    self.independent_base[i, j],
                    self.for_player_base[i, j],
                    self.true_base[i, j])
                objectives.append(objective_i_j)
        objectives_stack = torch.stack(objectives, dim=0)

        return objectives_stack.mean()

    def _binary_get_regrets(self):
        model = self.model
        B, n = self.for_player_reports.shape[0], self.players_num

        regrets_per_player = torch.zeros(n, device=self.device)

        if self.next_player_reports is None:
            honest_base = self.for_player_base.clone()
        else:
            honest_base = self.next_player_base.clone()

        misreport_base_0 = honest_base.clone()
        misreport_base_1 = honest_base.clone()

        for i in range(n):
            misreport_base_0[i, :, :, i] = 0
            misreport_base_1[i, :, :, i] = 1

        for i in range(n):
            regrets = []
            for j in range(B):
                u_truth = self.utility_fn(model, self.independent_base[i, j], self.for_player_base[i, j], i)
                u_0 = self.utility_fn(model, self.independent_base[i, j], misreport_base_0[i, j], i)
                u_1 = self.utility_fn(model, self.independent_base[i, j], misreport_base_1[i, j], i)

                u_mis = max(u_0.mean(), u_1.mean())
                regret = u_mis - u_truth.mean()
                regrets.append(regret)

            regrets = torch.stack(regrets, dim=0)
            mean_regret = regrets.mean()
            regrets_per_player[i] = mean_regret
        return regrets_per_player


class EPICEmpiricalLagrangianTrainer(EmpiricalLagrangianTrainer):
    """
    Trainer for training a EPIC approximating planner objective-maximising mechanism
    for selecting a winner.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def _set_sample(
            self,
            batch
    ):
        super()._set_sample(batch)
        # Prepare sample properly for other functions
        # Shape of reports: (B, n), first dimension is the batch dimension,
        # second is the player receiving report.
        # Shape of creted samples is (n, B, B, n)
        # First dimension is the player for whom the sample is used to calculate regret
        # The second dimension is from which batched sample we take the value of reports received by them
        # The third is batch dimension, fourth the player dimension of the tensor sent to mechanism.

        n, B = self.players_num, self.B
        self.true_base = (
            self.true_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.independent_base = (
            self.independent_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.for_player_base = (
            self.for_player_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )
        self.next_player_base = (
            self.next_player_reports.unsqueeze(0)
            .unsqueeze(2)
            .repeat(n, 1, B, 1)
            .clone()
        )

        if self.report_strategy == ReportStrategy.SELF_REPORT:
            for i in range(n):
                for j in range(B):
                    self.for_player_base[i, j, :, i] = self.for_player_reports[i, j]
                    self.independent_base[i, j, :, i] = self.independent_reports[i, j]
                    self.true_base[i, j, :, i] = self.true_reports[i, j]
        if self.report_strategy == ReportStrategy.OTHER_REPORT:
            for i in range(n):
                for j in range(B):
                    self.for_player_base[i, j, :, i] = self.for_player_reports[i, j]
                    self.independent_base[i, j, :, i] = self.independent_reports[i, j]
                    self.true_base[i, j, :, i] = self.true_reports[i, j]
                    self.next_player_base[i, j, :, i] = self.next_player_reports[i, j]

                    # Set next player
                    self.true_base[i, j, :, i + 1 % n] = self.true_reports[i + 1 % n, j]
                    self.for_player_base[i, j, :, i + 1 % n] = self.for_player_reports[i + 1 % n, j]
                    self.independent_base[i, j, :, i + 1 % n] = self.independent_reports[i + 1 % n, j]

                    # Set previous player
                    self.next_player_base[i, j, :, i - 1 % n] = self.next_player_reports[i - 1 % n, j]

    def _optimize_misreports(
            self,
            misreport_optimization_loops: int
    ) -> torch.Tensor:
        """
        For each sample in batch for each player calculates misreports maximising regret.
        A player misreport is a vector that maximizes interim regret:
        expected loss of utility when reporting truthfully, compared to when reporting optimally without knowledge
        of other players' reports (measured as expectation).
        :param misreport_optimization_loops:
        self.independent_reports: vector that has the independent reports, shape (B, n)
        self.for_player_reports: vector that has the reports for players, shape (B, n)
        :return: a tensor of size (B, n), for each player and batched game
        this is a misreport that maximises expected utility over all games where the player gets the value
        from the batched game.
        """
        B, n = self.for_player_reports.shape[0], self.players_num

        # Initialize misreports
        misreports_prime = torch.rand_like(self.for_player_reports, device=self.device, requires_grad=True)

        for _ in range(misreport_optimization_loops):
            if misreports_prime.grad is not None:
                misreports_prime.grad.detach_()
                misreports_prime.grad.zero_()

            if self.next_player_reports is None:
                reports_with_misreports = self.for_player_base.clone()
            else:
                reports_with_misreports = self.next_player_base.clone()
            # (n, B, B, n)

            # TODO Replace loops with vectorized operations
            for i in range(n):
                for j in range(B):
                    reports_with_misreports[i, j, :, i] = misreports_prime[j, i]

            utils = []

            for i in range(n):
                for j in range(B):
                    u_i_j = self.utility_fn(
                        self.model,
                        self.independent_base[i, j],  # shared (B, n)
                        reports_with_misreports[i, j],  # (B, n) for player i
                        i
                    )
                    utils.append(u_i_j)
            utils_stack = torch.stack(utils, dim=0)  # (n*B, B)

            scalar = utils_stack.sum()
            scalar.backward()

            with torch.no_grad():
                if misreports_prime.grad is None:
                    break
                misreports_prime += self.gamma * misreports_prime.grad
                misreports_prime.grad.zero_()

        return misreports_prime.detach()

    def _get_regrets_from_misreports(
            self,
            misreports_prime,
    ) -> torch.Tensor:
        B, n = self.for_player_reports.shape[0], self.players_num

        regrets_per_player = torch.zeros(n, device=self.device)

        if self.next_player_reports is None:
            honest_base = self.for_player_base.clone()
        else:
            honest_base = self.next_player_base.clone()

        misreport_base = honest_base.clone()

        for i in range(n):
            for j in range(B):
                misreport_base[i, j, :, i] = misreports_prime[i, j]

        for i in range(n):
            regrets = []
            for j in range(B):
                # reports for misreport case
                u_mis = self.utility_fn(self.model, self.independent_base[i, j], misreport_base[i, j], i)  # (B,)

                # truthful utility
                u_truth = self.utility_fn(self.model, self.independent_base[i, j], honest_base[i, j], i)  # (B,)

                regret = (u_mis - u_truth)  # (B,)
                regrets.append(regret)
            regrets = torch.stack(regrets, dim=0)
            mean_regret = regrets.mean()
            regrets_per_player[i] = mean_regret
        return regrets_per_player

    def _get_objective(
            self,
    ) -> torch.Tensor:
        objectives = []
        for i in range(self.players_num):
            for j in range(self.for_player_reports.shape[0]):
                objective_i_j = self.objective_fn(
                    self.model,
                    self.independent_base[i, j],
                    self.for_player_base[i, j],
                    self.true_base[i, j])
                objectives.append(objective_i_j)
        objectives_stack = torch.stack(objectives, dim=0)

        return objectives_stack.mean()

    def _binary_get_regrets(self):
        model = self.model
        B, n = self.for_player_reports.shape[0], self.players_num

        regrets_per_player = torch.zeros(n, device=self.device)

        if self.next_player_reports is None:
            honest_base = self.for_player_base.clone()
        else:
            honest_base = self.next_player_base.clone()

        misreport_base_0 = honest_base.clone()
        misreport_base_1 = honest_base.clone()

        for i in range(n):
            misreport_base_0[i, :, :, i] = 0
            misreport_base_1[i, :, :, i] = 1

        for i in range(n):
            regrets = []
            for j in range(B):
                u_truth = self.utility_fn(model, self.independent_base[i, j], self.for_player_base[i, j], i)
                u_0 = self.utility_fn(model, self.independent_base[i, j], misreport_base_0[i, j], i)
                u_1 = self.utility_fn(model, self.independent_base[i, j], misreport_base_1[i, j], i)

                u_mis = max(u_0.mean(), u_1.mean())
                regret = u_mis - u_truth.mean()
                regrets.append(regret)

            regrets = torch.stack(regrets, dim=0)
            mean_regret = regrets.mean()
            regrets_per_player[i] = mean_regret
        return regrets_per_player