# ETALK Counting

Official code repository for the paper:

## Minimal Embodiment Enables Efficient Learning of Number Concepts in Robot

![Paper Figure Placeholder](docs/figure_main.png)


## Overview

This project studies how minimal embodied signals help robots learn number concepts efficiently.
The repository contains three complementary model settings for controlled comparison:

- Embodied counting (visual + joint-state inputs)
- Single-image counting baseline
- Sequence-pooling baseline

## Repository Scope

This codebase provides:

- Model definitions and training pipelines
- Data loading utilities for each experimental setting
- Analysis scripts for representation and behavior-level evaluation
- Result extraction scripts for cross-experiment comparison

## Code Organization

- [Models](Models): neural network architectures
- [Data_loader](Data_loader): dataset and dataloader utilities
- [trainer.py](trainer.py), [trainer_single_image.py](trainer_single_image.py), [trainer_sequence_pooling.py](trainer_sequence_pooling.py): training logic
- [main.py](main.py), [main_single_image.py](main_single_image.py), [main_sequence_pooling.py](main_sequence_pooling.py): experiment entry points
- [analyze_embodied.py](analyze_embodied.py), [analyze_single_image.py](analyze_single_image.py): analysis and visualization
- [extract_all_results.py](extract_all_results.py): experiment summary aggregation

## Notes

- This repository is the research code used for the paper experiments.
- Paths and cluster-specific defaults may need adaptation for different environments.
- Non-English comments and strings were removed during open-source cleanup.

## System Requirements

### Hardware requirements

- CPU-only execution is supported for smoke tests and small-scale demos.
- GPU is strongly recommended for training and full analysis.
- Recommended GPU: NVIDIA GPU with at least 12 GB VRAM.

### Software requirements

- Python: 3.12.2
- OS: Linux (tested), macOS (expected to work), Windows (not officially tested)

Tested dependency versions (from conda environment `cgtest`):

- torch==2.5.1
- torchvision==0.20.1
- numpy==2.2.2
- pandas==2.2.3
- scikit-learn==1.6.1
- matplotlib==3.10.0
- seaborn==0.13.2
- Pillow==11.1.0
- opencv-python==4.12.0
- tqdm==4.67.1
- wandb==0.23.0

## Installation Guide

1. Clone the repository and enter the project folder.

```bash
git clone <your-repo-url>
cd ETALK_Counting
```

2. Activate your Python environment (recommended: conda environment used for experiments).

```bash
conda activate cgtest
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

Typical install time on a normal desktop computer:

- CPU-only environment: 5-15 minutes
- GPU-enabled environment with PyTorch/CUDA wheel setup: 10-30 minutes

## Demo

This repository does not include a toy dataset by default. The fastest verification path is a short run on your prepared dataset with reduced settings.

### Quick demo command (single-image baseline)

```bash
python main_single_image.py \
  --data_root /path/to/ball_data_collection \
  --train_csv_10 /path/to/ball_counting_dataset_train_10.csv \
  --val_csv /path/to/ball_counting_dataset_val.csv \
  --train_scales 10 \
  --seeds 2048 \
  --pretrain 1 \
  --total_epochs 1 \
  --batch_size 8 \
  --num_workers 2 \
  --save_dir ./experiments/demo_single_image
```

Expected output:

- Console logs showing training and validation progress for 1 epoch
- Checkpoint files under `./experiments/demo_single_image/.../checkpoints/`
- Training history JSON under the experiment output directory

Expected run time on a normal desktop computer:

- GPU: around 5-20 minutes (depends on dataset and I/O)
- CPU: can be substantially longer

## Instructions for Use

### 1. Prepare your data

The training scripts expect:

- A data root directory containing image files.
- CSV files with at least: `sample_id`, `ball_count`, `json_path`.
- JSON sequence files containing frame-wise image paths, labels, and (for embodied model) joint values.

Recommended in-repo layout:

```text
ETALK_Counting/
  data/
    ball_data_collection/
    Tools_script/
      ball_counting_dataset_train.csv
      ball_counting_dataset_train_50.csv
      ball_counting_dataset_train_10.csv
      ball_counting_dataset_val.csv
```

Download source for dataset files:

- OSF link: https://osf.io/jk4u8/overview?view_only=95fcde69554045788995b8ab2fdabc0d
- Download `ball_data_collection.zip` from the OSF project.

Place and extract the zip into this repository so that images end up under `data/ball_data_collection`:

```bash
# Run from ETALK_Counting/
mkdir -p data
unzip /path/to/ball_data_collection.zip -d data/
```

After extraction, confirm that `data/ball_data_collection` exists.

If your data is currently in external folders, migrate it into this project layout:

```bash
bash scripts/migrate_data_into_project.sh
```

You can also pass custom source paths:

```bash
bash scripts/migrate_data_into_project.sh /path/to/ball_data_collection /path/to/Tools_script
```

Important:

- Default script paths now point to `data/ball_data_collection` and `data/Tools_script`.
- You can still override paths with `--data_root`, `--train_csv_*`, and `--val_csv` if needed.

### 2. Run training

Embodied model (AlexNet + LSTM, visual + joints):

```bash
python main.py \
  --data_root /path/to/ball_data_collection \
  --train_csv_100 /path/to/ball_counting_dataset_train.csv \
  --train_csv_50 /path/to/ball_counting_dataset_train_50.csv \
  --train_csv_10 /path/to/ball_counting_dataset_train_10.csv \
  --val_csv /path/to/ball_counting_dataset_val.csv \
  --save_dir ./experiments/embodied
```

Single-image baseline:

```bash
python main_single_image.py \
  --data_root /path/to/ball_data_collection \
  --train_csv_100 /path/to/ball_counting_dataset_train.csv \
  --train_csv_50 /path/to/ball_counting_dataset_train_50.csv \
  --train_csv_10 /path/to/ball_counting_dataset_train_10.csv \
  --val_csv /path/to/ball_counting_dataset_val.csv \
  --save_dir ./experiments/single_image
```

Sequence-pooling baseline:

```bash
python main_sequence_pooling.py \
  --data_root /path/to/ball_data_collection \
  --train_csv_100 /path/to/ball_counting_dataset_train.csv \
  --train_csv_50 /path/to/ball_counting_dataset_train_50.csv \
  --train_csv_10 /path/to/ball_counting_dataset_train_10.csv \
  --val_csv /path/to/ball_counting_dataset_val.csv \
  --save_dir ./experiments/sequence_pooling
```

### 3. Reproduce the paper's main results

Recommended workflow:

1. Run all target training conditions for embodied, single-image, and sequence-pooling models.
2. Aggregate training summaries:

```bash
python extract_all_results.py
```

3. Run model-specific analysis scripts on selected checkpoints:

```bash
python analyze_embodied.py --checkpoint /path/to/best.pt --val_csv /path/to/val.csv --data_root /path/to/ball_data_collection
python analyze_single_image.py --checkpoint /path/to/best.pt --val_csv /path/to/val.csv --data_root /path/to/ball_data_collection
```

4. Optionally extract per-number accuracy trajectories:

```bash
python extract_per_number_accuracy_embodiment.py --experiments_dir /path/to/experiments --output_dir /path/to/output
python extract_per_number_accuracy_single_image.py --experiments_dir /path/to/experiments --output_dir /path/to/output
```

### Reproducibility notes

- Use fixed seeds (for example 2048 and 4096, as used by default).
- Keep preprocessing settings unchanged across model variants.
- Run all variants with the same train/validation splits.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for full text.

## Citation

If you use this code in academic work, please cite the paper:

```bibtex
@article{minimal_embodiment_number_concepts_robo,
  title={Minimal Embodiment Enables Efficient Learning of Number Concepts in Robot},
  author={Shangguan, Zhegong and Di Nuovo, Alessandro and Cangelosi, Angelo},
  journal={arXiv preprint arXiv:2604.11373},
  year={2026},
  url={https://arxiv.org/abs/2604.11373}
}
```
