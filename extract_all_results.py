#!/usr/bin/env python3
"""
Experiment results aggregation script.
Reads history.json files from experiment directories and produces summary CSVs and plots.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re


def parse_embodied_exp_name(exp_name):
    """
    Parse an Embodied Counting experiment name into a config dict.
    Example: EC_100pct_pre1_gate1_sj0_random_seed4096_20251129_195916
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Embodied'}
    
    # Extract data scale (e.g. '100pct' -> '100')
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # Extract boolean/int flags
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('gate'):
            config['use_modality_gate'] = bool(int(part[4]))
        elif part.startswith('sj'):
            config['shuffle_joints'] = bool(int(part[2]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # Extract curriculum mode
    if 'random' in exp_name:
        config['curriculum_mode'] = 'random'
    elif 'easy_to_hard' in exp_name:
        config['curriculum_mode'] = 'easy_to_hard'
    elif 'hard_to_easy' in exp_name:
        config['curriculum_mode'] = 'hard_to_easy'
    else:
        config['curriculum_mode'] = None
    
    # Extract timestamp suffix
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_single_image_exp_name(exp_name):
    """
    Parse a Single Image experiment name into a config dict.
    Example: SI_100pct_pre1_seed2048_20251202_123456
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Single_Image'}
    
    # Extract data scale
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # Extract boolean/int flags
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # Fields not applicable to this model type
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # Extract timestamp suffix
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_sequence_pooling_exp_name(exp_name):
    """
    Parse a Sequence Pooling experiment name into a config dict.
    Example: SP_100pct_pre1_poolmean_seed2048_20251204_001733
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Sequence_Pooling'}
    
    # Extract data scale
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # Extract flags and pooling strategy
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('pool'):
            config['pooling_strategy'] = part[4:]  # e.g. 'mean', 'max', 'last'
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # Fields not applicable to this model type
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # Extract timestamp suffix
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def extract_embodied_metrics(history):
    """Extract summary metrics from an Embodied model training history."""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # Find the epoch with the lowest validation loss
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_count_accuracy': float(best_epoch.get('val_count_accuracy', 0)),
        'best_val_final_count_accuracy': float(best_epoch.get('val_final_count_accuracy', 0)),
        'best_val_true_final_count_accuracy': float(best_epoch.get('val_true_final_count_accuracy', 0)),
        'best_val_joint_mse': float(best_epoch.get('val_joint_mse', 0)),
        
        # Training metrics at the best epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_count_accuracy_at_best': float(best_epoch.get('train_count_accuracy', 0)),
        
        # Metrics at the final epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_count_accuracy': float(df.iloc[-1].get('val_count_accuracy', 0)),
        'final_val_final_count_accuracy': float(df.iloc[-1].get('val_final_count_accuracy', 0)),
        'final_val_true_final_count_accuracy': float(df.iloc[-1].get('val_true_final_count_accuracy', 0)),
        'final_val_joint_mse': float(df.iloc[-1].get('val_joint_mse', 0)),
        
        # Peak metrics across all epochs
        'max_val_count_accuracy': float(df['val_count_accuracy'].max()),
        'max_val_final_count_accuracy': float(df['val_final_count_accuracy'].max()),
        'max_val_true_final_count_accuracy': float(df['val_true_final_count_accuracy'].max()),
        
        # Stability metrics
        'val_loss_std': float(df['val_loss'].std()),
        'val_count_accuracy_std': float(df['val_count_accuracy'].std()),
    }
    
    return metrics


def extract_single_image_metrics(history):
    """Extract summary metrics from a Single Image model training history."""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # Find the epoch with the lowest validation loss
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # Training metrics at the best epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # Metrics at the final epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # Peak accuracy across all epochs
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # Stability metrics
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # Alias fields for cross-model compatibility (mapped to val_accuracy / final_count)
        'best_val_count_accuracy': None,
        'best_val_final_count_accuracy': float(best_epoch.get('val_accuracy', 0)),
        'best_val_true_final_count_accuracy': float(best_epoch.get('val_accuracy', 0)),
        'best_val_joint_mse': None,
        'max_val_count_accuracy': None,
        'max_val_final_count_accuracy': float(df['val_accuracy'].max()),
        'max_val_true_final_count_accuracy': float(df['val_accuracy'].max()),
    }
    
    return metrics


def extract_sequence_pooling_metrics(history):
    """Extract summary metrics from a Sequence Pooling model training history."""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # Find the epoch with the lowest validation loss
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # Training metrics at the best epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # Metrics at the final epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # Peak accuracy across all epochs
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # Stability metrics
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # Alias fields for cross-model compatibility (mapped to val_accuracy / final_count)
        'best_val_count_accuracy': None,
        'best_val_final_count_accuracy': float(best_epoch.get('val_accuracy', 0)),
        'best_val_true_final_count_accuracy': float(best_epoch.get('val_accuracy', 0)),
        'best_val_joint_mse': None,
        'max_val_count_accuracy': None,
        'max_val_final_count_accuracy': float(df['val_accuracy'].max()),
        'max_val_true_final_count_accuracy': float(df['val_accuracy'].max()),
    }
    
    return metrics


def load_experiment_data(exp_dir, exp_name):
    """Load and parse a single experiment directory."""
    
    # Dispatch to the appropriate parser based on experiment prefix
    if exp_name.startswith('EC_'):
        config = parse_embodied_exp_name(exp_name)
        extract_metrics_fn = extract_embodied_metrics
    elif exp_name.startswith('SI_'):
        config = parse_single_image_exp_name(exp_name)
        extract_metrics_fn = extract_single_image_metrics
    elif exp_name.startswith('SP_'):
        config = parse_sequence_pooling_exp_name(exp_name)
        extract_metrics_fn = extract_sequence_pooling_metrics
    else:
        print(f"  Skipping unrecognised experiment prefix: {exp_name}")
        return None
    
    # Look for history.json
    history_path = os.path.join(exp_dir, 'history.json')
    if not os.path.exists(history_path):
        print(f"  Skipping — history.json not found: {exp_name}")
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Extract scalar metrics
        metrics = extract_metrics_fn(history)
        
        # Merge config and metrics
        result = {**config, **metrics}
        result['total_epochs'] = len(history)
        
        return result, history
    
    except Exception as e:
        print(f"  Failed to load {exp_name}: {e}")
        return None


def extract_all_experiments(experiments_dir):
    """Scan the experiments directory and load all recognised experiment runs.
    Each subdirectory starting with EC_, SI_, or SP_ is treated as one run.
    history.json is required; wandb logs are not used.
    """
    print("="*60)
    print("Scanning experiment directories...")
    print("="*60)

    all_results = []
    all_histories = {}

    if not os.path.exists(experiments_dir):
        print(f"Experiments directory not found: {experiments_dir}")
        return pd.DataFrame(), {}

    # Collect recognised experiment directories
    record_dirs = [p for p in Path(experiments_dir).iterdir()
                   if p.is_dir() and (p.name.startswith('EC_') or p.name.startswith('SI_') or p.name.startswith('SP_'))]

    print(f"\nFound {len(record_dirs)} experiment directories")

    embodied_count = 0
    single_image_count = 0
    sequence_pooling_count = 0
    processed = set()

    for record_dir in sorted(record_dirs, key=lambda x: x.name):
        exp_name = record_dir.name

        # Skip duplicates
        if exp_name in processed:
            continue

        # Find history.json — check the directory itself, then one level down
        candidate_dir = None
        if (record_dir / 'history.json').exists():
            candidate_dir = record_dir
        else:
            for child in record_dir.iterdir():
                if child.is_dir() and (child / 'history.json').exists():
                    candidate_dir = child
                    break

        if candidate_dir is None:
            print(f"  Skipping — history.json not found: {exp_name}")
            processed.add(exp_name)
            continue

        result = load_experiment_data(str(candidate_dir), exp_name)
        processed.add(exp_name)
        if result:
            data, history = result
            all_results.append(data)
            all_histories[data['exp_name']] = history

            if data['model_type'] == 'Embodied':
                embodied_count += 1
                print(f"  Loaded [Embodied] {data['exp_name']}")
            elif data['model_type'] == 'Single_Image':
                single_image_count += 1
                print(f"  Loaded [Single Image] {data['exp_name']}")
            elif data['model_type'] == 'Sequence_Pooling':
                sequence_pooling_count += 1
                print(f"  Loaded [Sequence Pooling] {data['exp_name']}")

    print(f"\nSummary:")
    print(f"  - Embodied: {embodied_count} experiments")
    print(f"  - Single Image: {single_image_count} experiments")
    print(f"  - Sequence Pooling: {sequence_pooling_count} experiments")
    print(f"  - Total: {len(all_results)} experiments")

    return pd.DataFrame(all_results), all_histories


def generate_summary_statistics(df):
    """Generate grouped summary statistics across all experiments."""
    print("\n" + "="*60)
    print("Generating summary statistics")
    print("="*60)
    
    # Per model type
    for model_type in df['model_type'].unique():
        print(f"\n{'='*60}")
        print(f"  {model_type} model statistics")
        print(f"{'='*60}")
        
        df_model = df[df['model_type'] == model_type]
        
        if model_type == 'Embodied':
            groupby_cols = ['data_scale', 'use_pretrain', 'use_modality_gate', 
                           'shuffle_joints', 'curriculum_mode']
            key_metric = 'best_val_true_final_count_accuracy'
        else:
            groupby_cols = ['data_scale', 'use_pretrain']
            key_metric = 'best_val_accuracy'
        
        # Only keep columns that exist in the dataframe
        groupby_cols = [col for col in groupby_cols if col in df_model.columns]
        
        if groupby_cols and key_metric in df_model.columns:
            agg_dict = {
                'best_val_loss': ['mean', 'std', 'min'],
                'best_epoch': ['mean', 'std', 'min', 'max'],
            }
            
            if model_type == 'Embodied':
                agg_dict.update({
                    'best_val_count_accuracy': ['mean', 'std', 'max'],
                    'best_val_final_count_accuracy': ['mean', 'std', 'max'],
                    'best_val_true_final_count_accuracy': ['mean', 'std', 'max'],
                })
            else:
                agg_dict.update({
                    'best_val_accuracy': ['mean', 'std', 'max'],
                })
            
            summary = df_model.groupby(groupby_cols).agg(agg_dict).round(4)
            print(summary)
    
    # Cross-model comparison
    print(f"\n{'='*60}")
    print("  Cross-model comparison by data scale")
    print(f"{'='*60}")
    
    comparison = df.groupby(['model_type', 'data_scale']).agg({
        'best_val_true_final_count_accuracy': ['mean', 'std', 'max'],
        'best_val_loss': ['mean', 'std', 'min'],
    }).round(4)
    print(comparison)
    
    return comparison


def plot_comparison_charts(df, save_dir):
    """Generate comparison bar/box plots across model types and configurations."""
    print("\n" + "="*60)
    print("Generating comparison charts...")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 1. Accuracy and loss by data scale, coloured by model type
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Embodied vs Single Image', fontsize=16)
    
    df_plot = df.copy()
    
    # Accuracy box plot
    ax = axes[0]
    sns.boxplot(data=df_plot, x='data_scale', y='best_val_true_final_count_accuracy', 
                hue='model_type', ax=ax)
    ax.set_title('Final Count Accuracy by Data Scale')
    ax.set_xlabel('Data Scale (%)')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model Type')
    
    # Loss box plot
    ax = axes[1]
    sns.boxplot(data=df_plot, x='data_scale', y='best_val_loss', 
                hue='model_type', ax=ax)
    ax.set_title('Validation Loss by Data Scale')
    ax.set_xlabel('Data Scale (%)')
    ax.set_ylabel('Loss')
    ax.legend(title='Model Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_comparison.png")
    
    # 2. Effect of pretraining on accuracy
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pretrain Effect Comparison', fontsize=16)
    
    for idx, model_type in enumerate(['Embodied', 'Single_Image']):
        df_model = df_plot[df_plot['model_type'] == model_type]
        ax = axes[idx]
        
        if not df_model.empty:
            sns.boxplot(data=df_model, x='use_pretrain', 
                       y='best_val_true_final_count_accuracy', ax=ax)
            ax.set_title(f'{model_type} Model')
            ax.set_xticklabels(['Random Init', 'Pretrained'])
            ax.set_ylabel('Final Count Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: pretrain_comparison.png")
    
    # 3. Per-scale comparison across model types
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance by Data Scale and Model Type', fontsize=16)
    
    scales = sorted(df_plot['data_scale'].unique())
    for idx, scale in enumerate(scales):
        df_scale = df_plot[df_plot['data_scale'] == scale]
        ax = axes[idx]
        
        sns.boxplot(data=df_scale, x='model_type', 
                   y='best_val_true_final_count_accuracy', ax=ax)
        ax.set_title(f'{scale}% Training Data')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Final Count Accuracy')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scale_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: scale_comparison.png")
    
    # 4. Embodied-specific: modality gate and curriculum learning effects
    df_embodied = df_plot[df_plot['model_type'] == 'Embodied']
    if not df_embodied.empty and 'use_modality_gate' in df_embodied.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Embodied Model: Gate and Curriculum Effects', fontsize=16)
        
        # Modality gate effect
        if df_embodied['use_modality_gate'].notna().any():
            sns.boxplot(data=df_embodied, x='use_modality_gate', 
                       y='best_val_true_final_count_accuracy', ax=axes[0])
            axes[0].set_title('Modality Gate Effect')
            axes[0].set_xticklabels(['No Gate', 'With Gate'])
            axes[0].set_ylabel('Final Count Accuracy')
        
        # Curriculum learning effect
        if df_embodied['curriculum_mode'].notna().any():
            sns.boxplot(data=df_embodied, x='curriculum_mode', 
                       y='best_val_true_final_count_accuracy', ax=axes[1])
            axes[1].set_title('Curriculum Learning Effect')
            axes[1].set_ylabel('Final Count Accuracy')
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'embodied_specific.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: embodied_specific.png")


def save_seed_aggregates(df, output_dir):
    """Aggregate metrics across seeds, grouping by config (ignoring seed, timestamp, exp_name).
    Outputs: by_config_seed_stats.csv
    """
    # Define numeric columns to aggregate
    numeric_cols = [
        'best_val_loss', 'best_epoch',
        'best_val_true_final_count_accuracy', 'best_val_final_count_accuracy', 'best_val_count_accuracy', 'best_val_accuracy',
        'val_loss_std', 'val_count_accuracy_std', 'val_accuracy_std',
        'final_val_loss', 'final_val_true_final_count_accuracy', 'final_val_final_count_accuracy', 'final_val_count_accuracy', 'final_val_accuracy',
        'max_val_true_final_count_accuracy', 'max_val_final_count_accuracy', 'max_val_count_accuracy', 'max_val_accuracy',
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        print("No numeric columns found for seed aggregation — skipping.")
        return None

    agg_spec = {c: ['mean', 'std', 'min', 'max'] for c in numeric_cols}

    pieces = []
    # Aggregate Embodied experiments
    if not df[df['model_type'] == 'Embodied'].empty:
        df_e = df[df['model_type'] == 'Embodied'].copy()
        group_cols_e = [c for c in ['model_type', 'data_scale', 'use_pretrain', 'use_modality_gate', 'shuffle_joints', 'curriculum_mode'] if c in df_e.columns]
        grouped_e = df_e.groupby(group_cols_e).agg(agg_spec)
        pieces.append(grouped_e)

    # Aggregate Single Image experiments
    if not df[df['model_type'] == 'Single_Image'].empty:
        df_s = df[df['model_type'] == 'Single_Image'].copy()
        group_cols_s = [c for c in ['model_type', 'data_scale', 'use_pretrain'] if c in df_s.columns]
        grouped_s = df_s.groupby(group_cols_s).agg(agg_spec)
        pieces.append(grouped_s)

    if not pieces:
        print("No experiments available for seed aggregation.")
        return None

    grouped = pd.concat(pieces, axis=0)
    grouped = grouped.round(4)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'by_config_seed_stats.csv')
    grouped.to_csv(out_path)
    print(f"Saved seed aggregates: {out_path}")
    return grouped


def save_seed2048_histories(df, histories, output_dir):
    """Save per-epoch training histories for seed=2048 experiments.
    One CSV per condition, named by the config string.
    """
    seed2048_dir = os.path.join(output_dir, 'seed2048_histories')
    os.makedirs(seed2048_dir, exist_ok=True)

    df_2048 = df[df['seed'] == 2048].copy()
    if df_2048.empty:
        print("No experiments found with seed=2048 — skipping.")
        return

    print(f"\nSaving seed=2048 training histories...")
    saved_count = 0

    for idx, row in df_2048.iterrows():
        exp_name = row['exp_name']
        if exp_name not in histories:
            continue

        # Build a descriptive condition name from config fields
        model_type = row['model_type']
        data_scale = row.get('data_scale', 'unknown')
        use_pretrain = row.get('use_pretrain', False)

        if model_type == 'Embodied':
            gate = row.get('use_modality_gate', False)
            sj = row.get('shuffle_joints', False)
            curr = row.get('curriculum_mode', 'none')
            condition_name = f"{model_type}_{data_scale}pct_pre{int(use_pretrain)}_gate{int(gate)}_sj{int(sj)}_{curr}"
        else:
            condition_name = f"{model_type}_{data_scale}pct_pre{int(use_pretrain)}"

        history_df = pd.DataFrame(histories[exp_name])
        out_path = os.path.join(seed2048_dir, f'{condition_name}.csv')
        history_df.to_csv(out_path, index=False)
        saved_count += 1
        print(f"  Saved {condition_name}.csv")

    print(f"Saved {saved_count} seed=2048 histories to: {seed2048_dir}")


def plot_learning_curves(histories, df, save_dir, top_n=5):
    """Plot training and validation loss/accuracy curves for the top N experiments."""
    print("\nGenerating learning curves...")
    
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        
        if df_model.empty:
            continue
        
        # Select top N experiments by primary metric
        sort_col = 'best_val_true_final_count_accuracy' if model_type == 'Embodied' else 'best_val_accuracy'
        if sort_col not in df_model.columns:
            continue
            
        top_exps = df_model.nlargest(top_n, sort_col)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Learning Curves - Top {top_n} {model_type} Experiments', fontsize=16)
        
        for _, row in top_exps.iterrows():
            exp_name = row['exp_name']
            if exp_name not in histories:
                continue
            
            history = pd.DataFrame(histories[exp_name])
            label = f"{row['data_scale']}% pre{int(row['use_pretrain'])} seed{row['seed']}"
            
            # Training loss
            axes[0, 0].plot(history['epoch'], history['train_loss'], label=label, alpha=0.7)
            
            # Validation loss
            axes[0, 1].plot(history['epoch'], history['val_loss'], label=label, alpha=0.7)
            
            # Accuracy curves
            if model_type == 'Embodied':
                axes[1, 0].plot(history['epoch'], history.get('train_count_accuracy', []), 
                               label=label, alpha=0.7)
                axes[1, 1].plot(history['epoch'], history.get('val_count_accuracy', []), 
                               label=label, alpha=0.7)
            else:
                axes[1, 0].plot(history['epoch'], history.get('train_accuracy', []), 
                               label=label, alpha=0.7)
                axes[1, 1].plot(history['epoch'], history.get('val_accuracy', []), 
                               label=label, alpha=0.7)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'learning_curves_{model_type.lower()}_top{top_n}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def main():
    """Main entry point — run full experiment aggregation and visualisation pipeline."""
    parser = argparse.ArgumentParser(description='Aggregate experiment results across EC/SI/SP runs')
    parser.add_argument('--experiments_dir', type=str, default=None,
                        help='Directory containing experiment run folders (default: <repo>/experiments)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save aggregation outputs (default: <repo>/analysis_results)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Experiment Results Aggregator")
    print("  Models: Embodied (EC_*) + Single Image (SI_*) + Sequence Pooling (SP_*)")
    print("="*60 + "\n")
    
    # Configure paths
    cur_dir = os.path.dirname(__file__)
    experiments_dir = args.experiments_dir or os.path.join(cur_dir, 'experiments')
    output_dir = args.output_dir or os.path.join(cur_dir, 'analysis_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiments
    df, histories = extract_all_experiments(experiments_dir)
    
    if df.empty:
        print("No experiments loaded — exiting.")
        return
    
    # Save combined CSV
    csv_path = os.path.join(output_dir, 'all_experiments_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved combined summary: {csv_path}")
    
    # Save per-model-type CSVs
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        model_type_clean = model_type.lower().replace(' ', '_')
        csv_model_path = os.path.join(output_dir, f'{model_type_clean}_experiments.csv')
        df_model.to_csv(csv_model_path, index=False)
        print(f"Saved {model_type} experiments: {csv_model_path}")
    
    # Generate and save grouped statistics
    summary = generate_summary_statistics(df)
    if summary is not None:
        summary_path = os.path.join(output_dir, 'comparison_statistics.csv')
        summary.to_csv(summary_path)
        print(f"Saved comparison statistics: {summary_path}")
    
    # Save seed-aggregated stats
    save_seed_aggregates(df, output_dir)

    # Save seed=2048 per-epoch histories
    save_seed2048_histories(df, histories, output_dir)
    
    # Print top-5 rankings per model type
    print("\n" + "="*60)
    print("Top experiments by model type")
    print("="*60)
    
    for model_type in ['Embodied', 'Single_Image', 'Sequence_Pooling']:
        df_model = df[df['model_type'] == model_type]
        if df_model.empty:
            continue
        
        print(f"\n{'='*60}")
        print(f"  {model_type} — Top 5")
        print(f"{'='*60}")
        
        sort_col = 'best_val_true_final_count_accuracy'
        top5 = df_model.nlargest(5, sort_col)
        
        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n#{rank} {row['exp_name']}")
            print(f"  Data scale: {row.get('data_scale', 'N/A')}%")
            print(f"  Pretrained: {row.get('use_pretrain', 'N/A')}")
            
            if model_type == 'Embodied':
                print(f"  Modality gate: {row.get('use_modality_gate', 'N/A')}")
                print(f"  Curriculum: {row.get('curriculum_mode', 'N/A')}")
                print(f"  True Final Accuracy: {row['best_val_true_final_count_accuracy']:.4f}")
                print(f"  Final Accuracy: {row.get('best_val_final_count_accuracy', 0):.4f}")
                print(f"  Count Accuracy: {row.get('best_val_count_accuracy', 0):.4f}")
                print(f"  Joint MSE: {row.get('best_val_joint_mse', 0):.6f}")
            elif model_type == 'Sequence_Pooling':
                print(f"  Pooling strategy: {row.get('pooling_strategy', 'N/A')}")
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (Final Count alias): {row['best_val_true_final_count_accuracy']:.4f}")
            else:  # Single_Image
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (Final Count alias): {row['best_val_true_final_count_accuracy']:.4f}")
            
            print(f"  Best epoch: {row['best_epoch']}")
            print(f"  Validation loss: {row['best_val_loss']:.4f}")
    
    print("\n" + "="*60)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print("="*60)
    print("\nOutput files:")
    print(f"  all_experiments_summary.csv     — all runs combined")
    print(f"  embodied_experiments.csv         — Embodied runs only")
    print(f"  single_image_experiments.csv     — Single Image runs only")
    print(f"  sequence_pooling_experiments.csv — Sequence Pooling runs only")
    print(f"  comparison_statistics.csv        — grouped statistics")
    print(f"  by_config_seed_stats.csv         — seed-aggregated statistics")
    print(f"  seed2048_histories/              — per-epoch histories for seed=2048")


if __name__ == '__main__':
    main()