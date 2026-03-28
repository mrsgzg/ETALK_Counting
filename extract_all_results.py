#!/usr/bin/env python3
"""
infoïinfoïinfo
info history.json info wandb info
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re


def parse_embodied_exp_name(exp_name):
    """
    info
    info: EC_100pct_pre1_gate1_sj0_random_seed4096_20251129_195916
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Embodied'}
    
    # info
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # info
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('gate'):
            config['use_modality_gate'] = bool(int(part[4]))
        elif part.startswith('sj'):
            config['shuffle_joints'] = bool(int(part[2]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # info
    if 'random' in exp_name:
        config['curriculum_mode'] = 'random'
    elif 'easy_to_hard' in exp_name:
        config['curriculum_mode'] = 'easy_to_hard'
    elif 'hard_to_easy' in exp_name:
        config['curriculum_mode'] = 'hard_to_easy'
    else:
        config['curriculum_mode'] = None
    
    # info
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_single_image_exp_name(exp_name):
    """
    info
    info: SI_100pct_pre1_seed2048_20251202_123456
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Single_Image'}
    
    # info
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # info
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # info
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # info
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_sequence_pooling_exp_name(exp_name):
    """
    info
    info: SP_100pct_pre1_poolmean_seed2048_20251204_001733
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Sequence_Pooling'}
    
    # info
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # info
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('pool'):
            config['pooling_strategy'] = part[4:]  # info'mean'info
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # info
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # info
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def extract_embodied_metrics(history):
    """history"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_count_accuracy': float(best_epoch.get('val_count_accuracy', 0)),
        'best_val_final_count_accuracy': float(best_epoch.get('val_final_count_accuracy', 0)),
        'best_val_true_final_count_accuracy': float(best_epoch.get('val_true_final_count_accuracy', 0)),
        'best_val_joint_mse': float(best_epoch.get('val_joint_mse', 0)),
        
        # epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_count_accuracy_at_best': float(best_epoch.get('train_count_accuracy', 0)),
        
        # epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_count_accuracy': float(df.iloc[-1].get('val_count_accuracy', 0)),
        'final_val_final_count_accuracy': float(df.iloc[-1].get('val_final_count_accuracy', 0)),
        'final_val_true_final_count_accuracy': float(df.iloc[-1].get('val_true_final_count_accuracy', 0)),
        'final_val_joint_mse': float(df.iloc[-1].get('val_joint_mse', 0)),
        
        # info
        'max_val_count_accuracy': float(df['val_count_accuracy'].max()),
        'max_val_final_count_accuracy': float(df['val_final_count_accuracy'].max()),
        'max_val_true_final_count_accuracy': float(df['val_true_final_count_accuracy'].max()),
        
        # info
        'val_loss_std': float(df['val_loss'].std()),
        'val_count_accuracy_std': float(df['val_count_accuracy'].std()),
    }
    
    return metrics


def extract_single_image_metrics(history):
    """history"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # info
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # info
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # infoïinfoïinfo- infoïfinal_count
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
    """history"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # epoch
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # epoch
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # info
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # info
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # infoïinfoïinfo- infoïfinal_count
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
    """info"""
    
    # info
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
        print(f"infoïinfo  info: {exp_name}")
        return None
    
    # history.json
    history_path = os.path.join(exp_dir, 'history.json')
    if not os.path.exists(history_path):
        print(f"infoïinfo  info history.json: {exp_name}")
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # info
        metrics = extract_metrics_fn(history)
        
        # info
        result = {**config, **metrics}
        result['total_epochs'] = len(history)
        
        return result, history
    
    except Exception as e:
        print(f"info info {exp_name}: {e}")
        return None


def extract_all_experiments(experiments_dir):
    """infoïinfoïinfo
    info experiments infoïinfo
    info history.jsonïinfoïinfo wandb info
    """
    print("="*60)
    print("ðinfo info...")
    print("="*60)

    all_results = []
    all_histories = {}

    if not os.path.exists(experiments_dir):
        print(f"info info: {experiments_dir}")
        return pd.DataFrame(), {}

    # info experiments info
    record_dirs = [p for p in Path(experiments_dir).iterdir()
                   if p.is_dir() and (p.name.startswith('EC_') or p.name.startswith('SI_') or p.name.startswith('SP_'))]

    print(f"\n {len(record_dirs)} info")

    embodied_count = 0
    single_image_count = 0
    sequence_pooling_count = 0
    processed = set()

    for record_dir in sorted(record_dirs, key=lambda x: x.name):
        exp_name = record_dir.name

        # infoïinfo
        if exp_name in processed:
            continue

        # info history.json
        candidate_dir = None
        if (record_dir / 'history.json').exists():
            candidate_dir = record_dir
        else:
            for child in record_dir.iterdir():
                if child.is_dir() and (child / 'history.json').exists():
                    candidate_dir = child
                    break

        if candidate_dir is None:
            print(f"infoïinfo  info history.json: {exp_name}")
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
                print(f"info [Embodied] {data['exp_name']}")
            elif data['model_type'] == 'Single_Image':
                single_image_count += 1
                print(f"info [Single Image] {data['exp_name']}")
            elif data['model_type'] == 'Sequence_Pooling':
                sequence_pooling_count += 1
                print(f"info [Sequence Pooling] {data['exp_name']}")

    print(f"\n:")
    print(f"  - info: {embodied_count} info")
    print(f"  - info: {single_image_count} info")
    print(f"  - info: {sequence_pooling_count} info")
    print(f"  - info: {len(all_results)} info")

    return pd.DataFrame(all_results), all_histories


def generate_summary_statistics(df):
    """info"""
    print("\n" + "="*60)
    print("ðinfo info")
    print("="*60)
    
    # info
    for model_type in df['model_type'].unique():
        print(f"\n{'='*60}")
        print(f"  {model_type} info")
        print(f"{'='*60}")
        
        df_model = df[df['model_type'] == model_type]
        
        if model_type == 'Embodied':
            groupby_cols = ['data_scale', 'use_pretrain', 'use_modality_gate', 
                           'shuffle_joints', 'curriculum_mode']
            key_metric = 'best_val_true_final_count_accuracy'
        else:  # Single_Image
            groupby_cols = ['data_scale', 'use_pretrain']
            key_metric = 'best_val_accuracy'
        
        # info
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
    
    # info
    print(f"\n{'='*60}")
    print("  infoïinfoïinfo")
    print(f"{'='*60}")
    
    comparison = df.groupby(['model_type', 'data_scale']).agg({
        'best_val_true_final_count_accuracy': ['mean', 'std', 'max'],
        'best_val_loss': ['mean', 'std', 'min'],
    }).round(4)
    print(comparison)
    
    return comparison


def plot_comparison_charts(df, save_dir):
    """info"""
    print("\n" + "="*60)
    print("ðinfo info...")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # info
    sns.set_style("whitegrid")
    
    # 1. infoïinfoïinfo
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Embodied vs Single Image', fontsize=16)
    
    # info - info
    df_plot = df.copy()
    
    # Accuracy info
    ax = axes[0]
    sns.boxplot(data=df_plot, x='data_scale', y='best_val_true_final_count_accuracy', 
                hue='model_type', ax=ax)
    ax.set_title('Final Count Accuracy by Data Scale')
    ax.set_xlabel('Data Scale (%)')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model Type')
    
    # Loss info
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
    print(f"info info: model_comparison.png")
    
    # 2. infoïinfoïinfo
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
    print(f"info info: pretrain_comparison.png")
    
    # 3. info
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
    print(f"info info: scale_comparison.png")
    
    # 4. infoïinfoïinfo
    df_embodied = df_plot[df_plot['model_type'] == 'Embodied']
    if not df_embodied.empty and 'use_modality_gate' in df_embodied.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Embodied Model: Gate and Curriculum Effects', fontsize=16)
        
        # info
        if df_embodied['use_modality_gate'].notna().any():
            sns.boxplot(data=df_embodied, x='use_modality_gate', 
                       y='best_val_true_final_count_accuracy', ax=axes[0])
            axes[0].set_title('Modality Gate Effect')
            axes[0].set_xticklabels(['No Gate', 'With Gate'])
            axes[0].set_ylabel('Final Count Accuracy')
        
        # info
        if df_embodied['curriculum_mode'].notna().any():
            sns.boxplot(data=df_embodied, x='curriculum_mode', 
                       y='best_val_true_final_count_accuracy', ax=axes[1])
            axes[1].set_title('Curriculum Learning Effect')
            axes[1].set_ylabel('Final Count Accuracy')
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'embodied_specific.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"info info: embodied_specific.png")


def save_seed_aggregates(df, output_dir):
    """infoïinfo seedtimestampexp_nameïinfo
    infoïinfo
    info: by_config_seed_stats.csv
    """
    # infoïinfoïinfo
    numeric_cols = [
        'best_val_loss', 'best_epoch',
        'best_val_true_final_count_accuracy', 'best_val_final_count_accuracy', 'best_val_count_accuracy', 'best_val_accuracy',
        'val_loss_std', 'val_count_accuracy_std', 'val_accuracy_std',
        'final_val_loss', 'final_val_true_final_count_accuracy', 'final_val_final_count_accuracy', 'final_val_count_accuracy', 'final_val_accuracy',
        'max_val_true_final_count_accuracy', 'max_val_final_count_accuracy', 'max_val_count_accuracy', 'max_val_accuracy',
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        print("infoïinfo infoïinfo")
        return None

    agg_spec = {c: ['mean', 'std', 'min', 'max'] for c in numeric_cols}

    pieces = []
    # Embodied info
    if not df[df['model_type'] == 'Embodied'].empty:
        df_e = df[df['model_type'] == 'Embodied'].copy()
        group_cols_e = [c for c in ['model_type', 'data_scale', 'use_pretrain', 'use_modality_gate', 'shuffle_joints', 'curriculum_mode'] if c in df_e.columns]
        grouped_e = df_e.groupby(group_cols_e).agg(agg_spec)
        pieces.append(grouped_e)

    # Single_Image info
    if not df[df['model_type'] == 'Single_Image'].empty:
        df_s = df[df['model_type'] == 'Single_Image'].copy()
        group_cols_s = [c for c in ['model_type', 'data_scale', 'use_pretrain'] if c in df_s.columns]
        grouped_s = df_s.groupby(group_cols_s).agg(agg_spec)
        pieces.append(grouped_s)

    if not pieces:
        print("infoïinfo info")
        return None

    grouped = pd.concat(pieces, axis=0)
    grouped = grouped.round(4)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'by_config_seed_stats.csv')
    grouped.to_csv(out_path)
    print(f"info info: {out_path}")
    return grouped


def save_seed2048_histories(df, histories, output_dir):
    """info seed=2048 info
    conditionïinfoïCSV
    """
    seed2048_dir = os.path.join(output_dir, 'seed2048_histories')
    os.makedirs(seed2048_dir, exist_ok=True)

    df_2048 = df[df['seed'] == 2048].copy()
    if df_2048.empty:
        print("infoïinfo info seed=2048 info")
        return

    print(f"\n seed=2048 info...")
    saved_count = 0

    for idx, row in df_2048.iterrows():
        exp_name = row['exp_name']
        if exp_name not in histories:
            continue

        # condition
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
        print(f"  info {condition_name}.csv")

    print(f"info info {saved_count} info seed=2048 info: {seed2048_dir}")


def plot_learning_curves(histories, df, save_dir, top_n=5):
    """info"""
    print("\nðinfo info...")
    
    # Top
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        
        if df_model.empty:
            continue
        
        # Top N
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
        print(f"info info: {filename}")


def main():
    """info"""
    print("\n" + "="*60)
    print("  info")
    print("  info: info (EC_*) + info (SI_*) + info (SP_*)")
    print("="*60 + "\n")
    
    # info
    base_dir = '/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Cognitive_Embodied_Counting'
    experiments_dir = os.path.join(base_dir, 'experiments')
    output_dir = os.path.join(base_dir, 'analysis_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # info
    df, histories = extract_all_experiments(experiments_dir)
    
    if df.empty:
        print("info infoïinfo")
        return
    
    # info
    csv_path = os.path.join(output_dir, 'all_experiments_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n info: {csv_path}")
    
    # info
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        # info
        model_type_clean = model_type.lower().replace(' ', '_')
        csv_model_path = os.path.join(output_dir, f'{model_type_clean}_experiments.csv')
        df_model.to_csv(csv_model_path, index=False)
        print(f"info info{model_type}info: {csv_model_path}")
    
    # info
    summary = generate_summary_statistics(df)
    if summary is not None:
        summary_path = os.path.join(output_dir, 'comparison_statistics.csv')
        summary.to_csv(summary_path)
        print(f"info info: {summary_path}")
    
    # seed
    save_seed_aggregates(df, output_dir)

    # info seed=2048 info
    save_seed2048_histories(df, histories, output_dir)
    
    # Top
    print("\n" + "="*60)
    print("ðinfo info")
    print("="*60)
    
    for model_type in ['Embodied', 'Single_Image', 'Sequence_Pooling']:
        df_model = df[df['model_type'] == model_type]
        if df_model.empty:
            continue
        
        print(f"\n{'='*60}")
        print(f"  {model_type} info - Top 5")
        print(f"{'='*60}")
        
        sort_col = 'best_val_true_final_count_accuracy'
        top5 = df_model.nlargest(5, sort_col)
        
        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n#{rank} {row['exp_name']}")
            print(f"  info: {row.get('data_scale', 'N/A')}%")
            print(f"  info: {row.get('use_pretrain', 'N/A')}")
            
            if model_type == 'Embodied':
                print(f"  info: {row.get('use_modality_gate', 'N/A')}")
                print(f"  info: {row.get('curriculum_mode', 'N/A')}")
                print(f"  True Final Accuracy: {row['best_val_true_final_count_accuracy']:.4f}")
                print(f"  Final Accuracy: {row.get('best_val_final_count_accuracy', 0):.4f}")
                print(f"  Count Accuracy: {row.get('best_val_count_accuracy', 0):.4f}")
                print(f"  Joint MSE: {row.get('best_val_joint_mse', 0):.6f}")
            elif model_type == 'Sequence_Pooling':
                print(f"  info: {row.get('pooling_strategy', 'N/A')}")
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (Final Count): {row['best_val_true_final_count_accuracy']:.4f}")
            else:  # Single_Image
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (Final Count): {row['best_val_true_final_count_accuracy']:.4f}")
            
            print(f"  Epoch: {row['best_epoch']}")
            print(f"  Validation Loss: {row['best_val_loss']:.4f}")
    
    print("\n" + "="*60)
    print(f"info info: {output_dir}")
    print("="*60)
    print("\n:")
    print(f"  ðinfo all_experiments_summary.csv - info")
    print(f"  ðinfo embodied_experiments.csv - info")
    print(f"  ðinfo single_image_experiments.csv - info")
    print(f"  ðinfo sequence_pooling_experiments.csv - info")
    print(f"  ðinfo comparison_statistics.csv - info")
    print(f"  ðinfo by_config_seed_stats.csv - info")
    print(f"  ðinfo seed2048_histories/ - seed=2048")


if __name__ == '__main__':
    main()
