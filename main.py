import argparse
from pathlib import Path

import torch

from Game import Model
from Game import DSICEmpiricalLagrangianTrainer, BICEmpiricalLagrangianTrainer
from Game.evaluation import ArgmaxMechanism
from Game.game import Game, ContinousErrorReportGeneration, CorrectReportGeneration, \
    ConstantChanceReportGeneration, InvertValueReportGeneration, BinaryGame
from Game.trainer import ReportStrategy
from Game.utility import objective_fn, utility_fn


REPORT_GENERATORS = {
    'correct': CorrectReportGeneration,
    'constant_chance': ConstantChanceReportGeneration,
    'invert_value': InvertValueReportGeneration,
    'continuous_error': ContinousErrorReportGeneration
}

TRAINER_GENERATORS = {
    'DSIC': DSICEmpiricalLagrangianTrainer,
    'BIC': BICEmpiricalLagrangianTrainer
}


def parse_args():
    parser = argparse.ArgumentParser(description="Configure the game theory / training environment.")

    # Integer parameters
    parser.add_argument('--train_dataset_size', type=int, default=2000, help='Total size of the training dataset')
    parser.add_argument('--eval_dataset_size', type=int, default=200, help='Total size of the evaluation dataset')
    parser.add_argument('--players_num', type=int, default=5, help='Number of players')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in the network')
    parser.add_argument('--epoch_num', type=int, default=100, help='Number of training epochs')

    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')

    # Distribution bounds
    parser.add_argument('--real_val_min', type=float, default=0.0, help='Min bound for real values distribution')
    parser.add_argument('--real_val_max', type=float, default=1.0, help='Max bound for real values distribution')

    # Player report generation
    parser.add_argument('--player_report_class', type=str, default='correct',
                        choices=list(REPORT_GENERATORS.keys()),
                        help="Which player report generation to use")
    parser.add_argument('--player_accuracy', type=float, default=0.8,
                        help="Player accuracy for constant_chance or invert_value strategies (0.0 to 1.0)")
    parser.add_argument('--player_err_min', type=float, default=-0.1)
    parser.add_argument('--player_err_max', type=float, default=0.1)

    # Independent report generation arguments
    parser.add_argument('--independent_report_class', type=str, default='correct',
                        choices=list(REPORT_GENERATORS.keys()),
                        help="Which independent report generation to use")
    parser.add_argument('--independent_accuracy', type=float, default=0.7,
                        help="Player accuracy for constant_chance or invert_value strategies (0.0 to 1.0)")
    parser.add_argument('--independent_err_min', type=float, default=-0.2)
    parser.add_argument('--independent_err_max', type=float, default=0.2)

    # Reporting strategy
    parser.add_argument(
        '--strategy',
        type=lambda s: s.upper(),
        default='SELF_REPORT',
        choices=[e.name for e in ReportStrategy],
        help="Reporting strategy to employ"
    )

    # DSIC or BIC
    parser.add_argument(
        '--compatibility',
        type=lambda s: s.upper(),
        default='DSIC',
        choices=list(TRAINER_GENERATORS.keys()),
        help="Compatibility criterion to employ"
    )

    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g., cpu, cuda)')

    # Model save path
    parser.add_argument(
        '--model_path',
        type=Path,
        default=Path("checkpoints/model.pt"),
        help="Path where the model checkpoint will be saved (e.g., checkpoints/model.pt)"
    )

    return parser.parse_args()


def extract_prefixed_args(args: argparse.Namespace, prefix: str) -> dict:
    """
    Extracts all arguments starting with `prefix` into a clean dictionary,
    removing the prefix from the keys.
    """
    args_dict = vars(args)

    return {
        key[len(prefix):]: value
        for key, value in args_dict.items()
        if key.startswith(prefix)
    }


def get_generator(args):
    generator_cls = REPORT_GENERATORS[args['report_class']]
    generator_kwargs = {}

    if generator_cls in (ConstantChanceReportGeneration, InvertValueReportGeneration):
        generator_kwargs['accuracy'] = args['accuracy']

    elif generator_cls is ContinousErrorReportGeneration:
        generator_kwargs['error_dist'] = torch.distributions.Uniform(
            args['err_min'],
            args['player_err_max']
        )

    report_generator = generator_cls(**generator_kwargs)
    return report_generator


def main(args):
    # Game setup
    players_num = args.players_num
    real_values_dist = torch.distributions.Uniform(args.real_val_min, args.real_val_max)

    report_strategy = ReportStrategy[args.strategy]

    independent_report_args = extract_prefixed_args(args, 'independent_')
    independent_report_generation = get_generator(independent_report_args)

    player_report_args = extract_prefixed_args(args, 'player_')
    for_player_report_generation = get_generator(player_report_args)

    # Training setup
    trainer_cls = TRAINER_GENERATORS[args.compatibility]
    train_dataset_size = args.train_dataset_size
    eval_dataset_size = args.eval_dataset_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    num_hidden_layers = args.num_hidden_layers
    epoch_num = args.epoch_num
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available():
        print("Training on GPU")
    else:
        device = torch.device('cpu')
        print("Training on CPU")
    rho_sched = lambda t: 1.0

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Model
    model = Model(players_num, num_hidden_layers).to(device)

    # Datasets
    train_dataset = Game(
        dataset_size=train_dataset_size,
        players_num=players_num,
        real_values_dist=real_values_dist,
        independent_report_generation=independent_report_generation,
        for_player_report_generation=for_player_report_generation
    )
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_dataset = Game(
        dataset_size=eval_dataset_size,
        players_num=players_num,
        real_values_dist=real_values_dist,
        independent_report_generation=independent_report_generation,
        for_player_report_generation=for_player_report_generation
    )
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True)

    trainer = trainer_cls(
        model=model,
        objective_fn=objective_fn,
        utility_fn=utility_fn,
        players_num=players_num,
        rho_scheduler=rho_sched,
        device=device,
        report_strategy=report_strategy,
    )
    trainer.train(train_dl, epoch_num)

    # Evaluation
    # Evaluation on actual case
    trainer.evaluate_model(model, eval_dl)
    trainer.evaluate_model(ArgmaxMechanism(players_num), eval_dl)

    # Evaluation on binary restricted case (player values restricted to 0 and 1).
    binary_eval_dataset = BinaryGame(
        dataset_size=eval_dataset_size,
        players_num=players_num,
        independent_report_generation=independent_report_generation,
        for_player_report_generation=for_player_report_generation
    )
    binary_eval_dl = torch.utils.data.DataLoader(binary_eval_dataset, batch_size=eval_batch_size, shuffle=True)
    trainer.binary_evaluate_model(model, binary_eval_dl)
    trainer.binary_evaluate_model(ArgmaxMechanism(players_num), binary_eval_dl)

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
