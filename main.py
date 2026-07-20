import torch

from Game import Model
from Game import EmpiricalLagrangianTrainer
from Game.evaluation import evaluate_model, ArgmaxMechanism
from Game.game import Game, ReportStrategy, ContinousErrorWrongReportGeneration
from Game.utility import objective_fn, utility_fn


def main():
    ### TODO These should be arguments:
    dataset_size = 2000
    players_num = 5

    real_values_dist = torch.distributions.Uniform(0.0, 1.0)
    independent_report_error_dist = torch.distributions.Uniform(-0.2, 0.2)
    player_report_error_dist = torch.distributions.Uniform(-0.2, 0.2)

    reportStrategy = ReportStrategy['SELF_REPORT']

    num_hidden_layers = 2
    rho_sched = lambda t: 1.0
    epoch_num = 100
    batch_size = 16
    device = "cpu"
    device = torch.device(device)
    # END TODO

    reportGeneration = ContinousErrorWrongReportGeneration(player_report_error_dist, independent_report_error_dist)
    dataset = Game(
        dataset_size=dataset_size,
        players_num=2,
        real_values_dist=real_values_dist,
        report_strategy=reportStrategy,
        report_generation=reportGeneration)

    model = Model(players_num, num_hidden_layers)
    trainer = EmpiricalLagrangianTrainer(
        model,
        objective_fn,
        utility_fn,
        players_num=players_num,
        rho_scheduler=rho_sched
    )
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainer.train(dl, epoch_num)

    evaluate_model(model, dl, objective_fn, utility_fn, players_num, device)
    evaluate_model(ArgmaxMechanism(5), dl, objective_fn, utility_fn, players_num, device)
    torch.save(model.state_dict(), "trained_model.pth")


if __name__ == "__main__":
    pass
