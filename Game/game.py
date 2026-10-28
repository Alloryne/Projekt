import abc
from abc import abstractmethod
from Game.trainer import ReportStrategy

import torch
from torch.utils.data import Dataset


class AbstractReportGeneration(abc.ABC):
    @abstractmethod
    def get_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(true_values)


class CorrectReportGeneration(AbstractReportGeneration):
    def get_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        return true_values.clone()


class ConstantChanceReportGeneration(AbstractReportGeneration):
    def __init__(self, accuracy, **kwargs):
        self.accuracy = accuracy

    def _get_false_values(self, true_values: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(true_values)

    def get_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        false_values = self._get_false_values(true_values)
        flip = torch.rand_like(true_values) < (1.0 - self.accuracy)
        reports = torch.where(flip, false_values, true_values)
        return reports


class InvertValueReportGeneration(ConstantChanceReportGeneration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_false_values(self, true_values: torch.Tensor) -> torch.Tensor:
        return 1.0 - true_values


class ContinousErrorReportGeneration(AbstractReportGeneration):
    def __init__(self,
                 error_dist: torch.distributions.Distribution,
                 **kwargs):
        super().__init__(kwargs)
        self.error_dist = error_dist

    def get_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        errors = self.error_dist.sample(true_values.shape)
        reports = true_values + errors
        return reports


class Game(Dataset):
    """
    Abstract Dataset that represents a game.
    It stores the number of players, distribution of player valuations.
    It returns the true valuations, valuations reported to the players
    and independent reports of valuations.
    """
    def __init__(self, dataset_size: int, players_num: int,
                 real_values_dist: torch.distributions.Distribution,
                 independent_report_generation: AbstractReportGeneration,
                 for_player_report_generation: AbstractReportGeneration,
                 report_strategy: ReportStrategy = ReportStrategy.SELF_REPORT,
                 for_next_player_report_generation: AbstractReportGeneration = CorrectReportGeneration()):
        self.dataset_size = dataset_size
        self.players_num = players_num
        self.real_values_dist = real_values_dist
        self.for_player_report_generation = for_player_report_generation
        self.independent_report_generation = independent_report_generation
        self.report_strategy = report_strategy
        self.for_next_player_report_generation = (
            for_next_player_report_generation
        )

        self._generate_data()

    def _generate_data(self):
        self.true_values = self.real_values_dist.sample(
            (self.dataset_size, self.players_num)
        )
        self.for_player_reports = self.for_player_report_generation.get_reports(self.true_values)
        self.independent_reports = self.independent_report_generation.get_reports(self.true_values)
        if self.report_strategy == ReportStrategy.OTHER_REPORT:
            next_player_values = torch.roll(
                self.true_values, shifts=-1, dims=-1
            )
            self.for_next_player_reports = (
                self.for_next_player_report_generation.get_reports(
                    next_player_values
                )
            )

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int):
        if self.report_strategy == ReportStrategy.OTHER_REPORT:
            return {
                "true_values": self.true_values[idx],
                "for_player_reports": self.for_player_reports[idx],
                "for_next_player_reports": self.for_next_player_reports[idx],
                "independent_reports": self.independent_reports[idx],
            }

        return {
            "true_values": self.true_values[idx],
            "for_player_reports": self.for_player_reports[idx],
            "independent_reports": self.independent_reports[idx],
        }


class BinaryGame(Game):
    """
    Abstract Dataset that represents a game.
    It stores the number of players, distribution of player valuations.
    It returns the true valuations, valuations reported to the players
    and independent reports of valuations.
    """
    def __init__(self, dataset_size: int, players_num: int,
                 independent_report_generation: AbstractReportGeneration,
                 for_player_report_generation: AbstractReportGeneration,
                 report_strategy: ReportStrategy = ReportStrategy.SELF_REPORT,
                 for_next_player_report_generation: AbstractReportGeneration = CorrectReportGeneration()):
        # TODO FIX FOR CONTINOUS REPORT GENERATION
        super().__init__(dataset_size, players_num, torch.distributions.Bernoulli(probs=0.5),
                         independent_report_generation,
                         for_player_report_generation,
                         report_strategy,
                         for_next_player_report_generation)
