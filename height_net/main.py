import argparse
from simple_baseline_experiment_runner import SimpleBaselineExperimentRunner


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load CelebA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--model_path', type=str, help='Model saving path. eg: vggWith3LinearLayers', required=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--log_validation', action='store_true')
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    #elif args.model == "coattention":
    #    experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_data_loader_workers=args.num_data_loader_workers,
            model_path=args.model_path,
            lr=args.lr,
            log_validation=args.log_validation
    )
    experiment_runner.train()
