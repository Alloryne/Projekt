import abc
from abc import abstractmethod
from enum import Enum

import torch


class ReportStrategy(Enum):
    SELF_REPORT = 1
    OTHER_REPORT = 2


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


class AbstractWrongReportGeneration(abc.ABC):
    @abstractmethod
    def get_for_player_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(true_values)

    @abstractmethod
    def get_independent_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(true_values)


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


class ConstantChanceWrongReportGeneration(AbstractWrongReportGeneration):
    def __init__(self, for_player_report_accuracy, independent_report_accuracy, **kwargs):
        self.for_player_report_accuracy = for_player_report_accuracy
        self.independent_report_accuracy = independent_report_accuracy

    def _get_false_values(self, true_values: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(true_values)

    def get_for_player_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        false_values = self._get_false_values(true_values)
        flip = torch.rand_like(true_values) < (1.0 - self.for_player_report_accuracy)
        reports = torch.where(flip, false_values, true_values)
        return reports

    def get_independent_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        inverted_values = self._get_false_values(true_values)
        flip = torch.rand_like(true_values) < (1.0 - self.independent_report_accuracy)
        reports = torch.where(flip, inverted_values, true_values)
        return reports


class InvertValueWrongReportGeneration(ConstantChanceWrongReportGeneration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_false_values(self, true_values: torch.Tensor) -> torch.Tensor:
        return 1.0 - true_values


class ContinousErrorWrongReportGeneration(AbstractWrongReportGeneration):
    def __init__(self,
                 for_player_report_error_dist: torch.distributions.Distribution,
                 independent_report_error_dist: torch.distributions.Distribution,
                 **kwargs):
        super().__init__(kwargs)
        self.for_player_report_error_dist = for_player_report_error_dist
        self.independent_report_error_dist = independent_report_error_dist

    def get_for_player_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        errors = self.for_player_report_error_dist.sample(true_values.shape)
        reports = true_values + errors
        return reports

    def get_independent_reports(self, true_values: torch.Tensor) -> torch.Tensor:
        errors = self.independent_report_error_dist.sample(true_values.shape)
        reports = true_values + errors
        return reports


class Game(torch.utils.data.Dataset):
    """
    Abstract Dataset that represents a game.
    It stores the number of players, distribution of player valuations.
    It returns the true valuations, valuations reported to the players
    and independent reports of valuations.
    """
    def __init__(self, dataset_size: int, players_num: int,
                 real_values_dist: torch.distributions.Distribution,
                 report_strategy: ReportStrategy,
                 independent_report_generation: AbstractReportGeneration,
                 for_player_report_generation: AbstractReportGeneration):
        self.dataset_size = dataset_size
        self.players_num = players_num
        self.real_values_dist = real_values_dist
        self.report_strategy = report_strategy
        self.report_generation = independent_report_generation
        self.independent_report_generation = for_player_report_generation

    def __getitem__(self, idx):
        true_values = self.real_values_dist.sample((self.players_num,))
        for_player_reports = self.report_generation.get_reports(true_values)
        independent_reports = self.independent_report_generation.get_reports(true_values)

        if self.report_strategy == ReportStrategy.OTHER_REPORT:
            for_player_reports = torch.roll(for_player_reports, shifts=1, dims=0)

        return {
            'true_values': true_values,
            'for_players_reports': for_player_reports,
            'independent_reports': independent_reports
        }
