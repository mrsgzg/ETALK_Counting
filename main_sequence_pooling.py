import os
import argparse
from datetime import datetime

from trainer_sequence_pooling import run_training_once


def main():
    parser = argparse.ArgumentParser(description='Sequence Pooling Classification Experiments (wandb)')

    # data paths
    parser.add_argument('--data_root', type=str,
                        default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection')
    parser.add_argument('--train_csv_100', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv')
    parser.add_argument('--train_csv_50', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_50.csv')
    parser.add_argument('--train_csv_10', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv')
    parser.add_argument('--val_csv', type=str,
                        default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv')

    # training
    parser.add_argument('--total_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sequence_length', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # experiment grid
    parser.add_argument('--seeds', nargs='+', type=int, default=[2048, 4096])
    parser.add_argument('--pretrain', nargs='+', type=int, default=[1, 0], help='1 for True, 0 for False')
    parser.add_argument('--pooling_strategies', nargs='+', default=['mean', 'max', 'last'],
                        help='Pooling strategies: mean|max|last')
    parser.add_argument('--train_scales', nargs='+', default=['100', '50', '10'],
                        help='Which train csv set to use: 100|50|10')

    # wandb
    parser.add_argument('--project', type=str, default='sequence-pooling-counting')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional base directory to store experiments; default is experiments/<exp_name>')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Resolve which train csv to use per scale
    csv_map = {
        '100': args.train_csv_100,
        '50': args.train_csv_50,
        '10': args.train_csv_10,
    }

    total_runs = 0
    for _ in args.seeds:
        for _ in args.pretrain:
            for _ in args.pooling_strategies:
                for _ in args.train_scales:
                    total_runs += 1

    run_idx = 0
    for seed in args.seeds:
        for pre in args.pretrain:
            for pool in args.pooling_strategies:
                for scale in args.train_scales:
                    run_idx += 1
                    train_csv = csv_map[scale]
                    exp_name = f"SP_{scale}pct_pre{pre}_pool{pool}_seed{seed}_{timestamp}"
                    print(f"\n[Run {run_idx}/{total_runs}] {exp_name}")

                    run_training_once(
                        exp_name=exp_name,
                        data_root=args.data_root,
                        train_csv=train_csv,
                        val_csv=args.val_csv,
                        total_epochs=args.total_epochs,
                        batch_size=args.batch_size,
                        sequence_length=args.sequence_length,
                        num_workers=args.num_workers,
                        learning_rate=args.learning_rate,
                        use_pretrain=bool(pre),
                        pooling_strategy=pool,
                        seed=seed,
                        project=args.project,
                        save_dir=args.save_dir,
                    )


if __name__ == '__main__':
    main()
