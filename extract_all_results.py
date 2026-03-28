#!/usr/bin/env python3
"""
提取所有实验的训练结果（包括具身模型和单图像模型）
从 history.json 和 wandb 日志中提取关键指标
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
    解析具身模型实验名称
    例如: EC_100pct_pre1_gate1_sj0_random_seed4096_20251129_195916
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Embodied'}
    
    # 提取数据比例
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # 提取配置
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('gate'):
            config['use_modality_gate'] = bool(int(part[4]))
        elif part.startswith('sj'):
            config['shuffle_joints'] = bool(int(part[2]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # 提取课程学习模式
    if 'random' in exp_name:
        config['curriculum_mode'] = 'random'
    elif 'easy_to_hard' in exp_name:
        config['curriculum_mode'] = 'easy_to_hard'
    elif 'hard_to_easy' in exp_name:
        config['curriculum_mode'] = 'hard_to_easy'
    else:
        config['curriculum_mode'] = None
    
    # 提取时间戳
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_single_image_exp_name(exp_name):
    """
    解析单图像模型实验名称
    例如: SI_100pct_pre1_seed2048_20251202_123456
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Single_Image'}
    
    # 提取数据比例
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # 提取配置
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # 单图像模型没有这些配置
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # 提取时间戳
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def parse_sequence_pooling_exp_name(exp_name):
    """
    解析序列池化模型实验名称
    例如: SP_100pct_pre1_poolmean_seed2048_20251204_001733
    """
    parts = exp_name.split('_')
    config = {'model_type': 'Sequence_Pooling'}
    
    # 提取数据比例
    for i, part in enumerate(parts):
        if 'pct' in part:
            config['data_scale'] = part.replace('pct', '')
            break
    
    # 提取配置
    for i, part in enumerate(parts):
        if part.startswith('pre'):
            config['use_pretrain'] = bool(int(part[3]))
        elif part.startswith('pool'):
            config['pooling_strategy'] = part[4:]  # 提取'mean'等
        elif part.startswith('seed'):
            config['seed'] = int(part[4:])
    
    # 序列池化模型没有这些配置
    config['use_modality_gate'] = None
    config['shuffle_joints'] = None
    config['curriculum_mode'] = None
    
    # 提取时间戳
    timestamp_match = re.search(r'(\d{8}_\d{6})$', exp_name)
    if timestamp_match:
        config['timestamp'] = timestamp_match.group(1)
    
    config['exp_name'] = exp_name
    
    return config


def extract_embodied_metrics(history):
    """从具身模型history中提取指标"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # 找到验证损失最低的epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_count_accuracy': float(best_epoch.get('val_count_accuracy', 0)),
        'best_val_final_count_accuracy': float(best_epoch.get('val_final_count_accuracy', 0)),
        'best_val_true_final_count_accuracy': float(best_epoch.get('val_true_final_count_accuracy', 0)),
        'best_val_joint_mse': float(best_epoch.get('val_joint_mse', 0)),
        
        # 在最佳epoch时的训练指标
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_count_accuracy_at_best': float(best_epoch.get('train_count_accuracy', 0)),
        
        # 最终epoch的指标
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_count_accuracy': float(df.iloc[-1].get('val_count_accuracy', 0)),
        'final_val_final_count_accuracy': float(df.iloc[-1].get('val_final_count_accuracy', 0)),
        'final_val_true_final_count_accuracy': float(df.iloc[-1].get('val_true_final_count_accuracy', 0)),
        'final_val_joint_mse': float(df.iloc[-1].get('val_joint_mse', 0)),
        
        # 最大准确率
        'max_val_count_accuracy': float(df['val_count_accuracy'].max()),
        'max_val_final_count_accuracy': float(df['val_final_count_accuracy'].max()),
        'max_val_true_final_count_accuracy': float(df['val_true_final_count_accuracy'].max()),
        
        # 训练稳定性指标
        'val_loss_std': float(df['val_loss'].std()),
        'val_count_accuracy_std': float(df['val_count_accuracy'].std()),
    }
    
    return metrics


def extract_single_image_metrics(history):
    """从单图像模型history中提取指标"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # 找到验证损失最低的epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # 在最佳epoch时的训练指标
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # 最终epoch的指标
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # 最大准确率
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # 训练稳定性指标
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # 对齐字段（用于对比）- 单图像只看最终帧，相当于final_count
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
    """从序列池化模型history中提取指标"""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # 找到验证损失最低的epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    metrics = {
        'best_epoch': int(best_epoch['epoch']),
        'best_val_loss': float(best_epoch['val_loss']),
        'best_val_accuracy': float(best_epoch.get('val_accuracy', 0)),
        
        # 在最佳epoch时的训练指标
        'train_loss_at_best': float(best_epoch.get('train_loss', 0)),
        'train_accuracy_at_best': float(best_epoch.get('train_accuracy', 0)),
        
        # 最终epoch的指标
        'final_epoch': int(df.iloc[-1]['epoch']),
        'final_val_loss': float(df.iloc[-1]['val_loss']),
        'final_val_accuracy': float(df.iloc[-1].get('val_accuracy', 0)),
        
        # 最大准确率
        'max_val_accuracy': float(df['val_accuracy'].max()),
        
        # 训练稳定性指标
        'val_loss_std': float(df['val_loss'].std()),
        'val_accuracy_std': float(df['val_accuracy'].std()),
        
        # 对齐字段（用于对比）- 序列池化是单次分类，相当于final_count
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
    """加载单个实验的数据"""
    
    # 判断模型类型并解析配置
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
        print(f"⚠️  未知实验类型: {exp_name}")
        return None
    
    # 加载history.json
    history_path = os.path.join(exp_dir, 'history.json')
    if not os.path.exists(history_path):
        print(f"⚠️  未找到 history.json: {exp_name}")
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 提取指标
        metrics = extract_metrics_fn(history)
        
        # 合并配置和指标
        result = {**config, **metrics}
        result['total_epochs'] = len(history)
        
        return result, history
    
    except Exception as e:
        print(f"❌ 加载失败 {exp_name}: {e}")
        return None


def extract_all_experiments(experiments_dir):
    """提取所有实验的数据（包括具身和单图像）
    仅扫描 experiments 目录下的一级子目录作为实验记录目录；
    在记录目录本身或其下一级子目录中查找 history.json，不再更深层搜索，避免 wandb 目录造成重复。
    """
    print("="*60)
    print("🔍 开始提取实验数据...")
    print("="*60)

    all_results = []
    all_histories = {}

    if not os.path.exists(experiments_dir):
        print(f"❌ 实验目录不存在: {experiments_dir}")
        return pd.DataFrame(), {}

    # 仅扫描 experiments 的下一级目录
    record_dirs = [p for p in Path(experiments_dir).iterdir()
                   if p.is_dir() and (p.name.startswith('EC_') or p.name.startswith('SI_') or p.name.startswith('SP_'))]

    print(f"\n找到 {len(record_dirs)} 个实验目录")

    embodied_count = 0
    single_image_count = 0
    sequence_pooling_count = 0
    processed = set()

    for record_dir in sorted(record_dirs, key=lambda x: x.name):
        exp_name = record_dir.name

        # 去重：同名记录只处理一次
        if exp_name in processed:
            continue

        # 在记录目录或其下一级子目录查找 history.json
        candidate_dir = None
        if (record_dir / 'history.json').exists():
            candidate_dir = record_dir
        else:
            for child in record_dir.iterdir():
                if child.is_dir() and (child / 'history.json').exists():
                    candidate_dir = child
                    break

        if candidate_dir is None:
            print(f"⚠️  未找到 history.json: {exp_name}")
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
                print(f"✓ [Embodied] {data['exp_name']}")
            elif data['model_type'] == 'Single_Image':
                single_image_count += 1
                print(f"✓ [Single Image] {data['exp_name']}")
            elif data['model_type'] == 'Sequence_Pooling':
                sequence_pooling_count += 1
                print(f"✓ [Sequence Pooling] {data['exp_name']}")

    print(f"\n成功加载:")
    print(f"  - 具身模型: {embodied_count} 个实验")
    print(f"  - 单图像模型: {single_image_count} 个实验")
    print(f"  - 序列池化模型: {sequence_pooling_count} 个实验")
    print(f"  - 总计: {len(all_results)} 个实验")

    return pd.DataFrame(all_results), all_histories


def generate_summary_statistics(df):
    """生成统计摘要"""
    print("\n" + "="*60)
    print("📊 实验统计摘要")
    print("="*60)
    
    # 按模型类型分别统计
    for model_type in df['model_type'].unique():
        print(f"\n{'='*60}")
        print(f"  {model_type} 模型统计")
        print(f"{'='*60}")
        
        df_model = df[df['model_type'] == model_type]
        
        if model_type == 'Embodied':
            groupby_cols = ['data_scale', 'use_pretrain', 'use_modality_gate', 
                           'shuffle_joints', 'curriculum_mode']
            key_metric = 'best_val_true_final_count_accuracy'
        else:  # Single_Image
            groupby_cols = ['data_scale', 'use_pretrain']
            key_metric = 'best_val_accuracy'
        
        # 过滤存在的列
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
    
    # 整体对比
    print(f"\n{'='*60}")
    print("  模型类型整体对比（按数据规模）")
    print(f"{'='*60}")
    
    comparison = df.groupby(['model_type', 'data_scale']).agg({
        'best_val_true_final_count_accuracy': ['mean', 'std', 'max'],
        'best_val_loss': ['mean', 'std', 'min'],
    }).round(4)
    print(comparison)
    
    return comparison


def plot_comparison_charts(df, save_dir):
    """生成对比图表"""
    print("\n" + "="*60)
    print("📈 生成可视化图表...")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 1. 模型类型对比（按数据规模）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Embodied vs Single Image', fontsize=16)
    
    # 准备数据 - 使用统一的指标
    df_plot = df.copy()
    
    # Accuracy 对比
    ax = axes[0]
    sns.boxplot(data=df_plot, x='data_scale', y='best_val_true_final_count_accuracy', 
                hue='model_type', ax=ax)
    ax.set_title('Final Count Accuracy by Data Scale')
    ax.set_xlabel('Data Scale (%)')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model Type')
    
    # Loss 对比
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
    print(f"✓ 保存: model_comparison.png")
    
    # 2. 预训练效果对比（分模型类型）
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
    print(f"✓ 保存: pretrain_comparison.png")
    
    # 3. 按数据规模的详细对比
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
    print(f"✓ 保存: scale_comparison.png")
    
    # 4. 具身模型特有的对比（如果有数据）
    df_embodied = df_plot[df_plot['model_type'] == 'Embodied']
    if not df_embodied.empty and 'use_modality_gate' in df_embodied.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Embodied Model: Gate and Curriculum Effects', fontsize=16)
        
        # 模态门控效果
        if df_embodied['use_modality_gate'].notna().any():
            sns.boxplot(data=df_embodied, x='use_modality_gate', 
                       y='best_val_true_final_count_accuracy', ax=axes[0])
            axes[0].set_title('Modality Gate Effect')
            axes[0].set_xticklabels(['No Gate', 'With Gate'])
            axes[0].set_ylabel('Final Count Accuracy')
        
        # 课程学习效果
        if df_embodied['curriculum_mode'].notna().any():
            sns.boxplot(data=df_embodied, x='curriculum_mode', 
                       y='best_val_true_final_count_accuracy', ax=axes[1])
            axes[1].set_title('Curriculum Learning Effect')
            axes[1].set_ylabel('Final Count Accuracy')
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'embodied_specific.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: embodied_specific.png")


def save_seed_aggregates(df, output_dir):
    """按配置（不含 seed、timestamp、exp_name）聚合同一实验不同随机种子的均值与标准差。
    为避免单图像被具身专用列过滤，分别按模型类型分组再合并。
    生成文件: by_config_seed_stats.csv
    """
    # 可聚合的数值列（存在才用）
    numeric_cols = [
        'best_val_loss', 'best_epoch',
        'best_val_true_final_count_accuracy', 'best_val_final_count_accuracy', 'best_val_count_accuracy', 'best_val_accuracy',
        'val_loss_std', 'val_count_accuracy_std', 'val_accuracy_std',
        'final_val_loss', 'final_val_true_final_count_accuracy', 'final_val_final_count_accuracy', 'final_val_count_accuracy', 'final_val_accuracy',
        'max_val_true_final_count_accuracy', 'max_val_final_count_accuracy', 'max_val_count_accuracy', 'max_val_accuracy',
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        print("⚠️ 聚合失败：缺少数值列")
        return None

    agg_spec = {c: ['mean', 'std', 'min', 'max'] for c in numeric_cols}

    pieces = []
    # Embodied 分组
    if not df[df['model_type'] == 'Embodied'].empty:
        df_e = df[df['model_type'] == 'Embodied'].copy()
        group_cols_e = [c for c in ['model_type', 'data_scale', 'use_pretrain', 'use_modality_gate', 'shuffle_joints', 'curriculum_mode'] if c in df_e.columns]
        grouped_e = df_e.groupby(group_cols_e).agg(agg_spec)
        pieces.append(grouped_e)

    # Single_Image 分组
    if not df[df['model_type'] == 'Single_Image'].empty:
        df_s = df[df['model_type'] == 'Single_Image'].copy()
        group_cols_s = [c for c in ['model_type', 'data_scale', 'use_pretrain'] if c in df_s.columns]
        grouped_s = df_s.groupby(group_cols_s).agg(agg_spec)
        pieces.append(grouped_s)

    if not pieces:
        print("⚠️ 没有可聚合的数据")
        return None

    grouped = pd.concat(pieces, axis=0)
    grouped = grouped.round(4)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'by_config_seed_stats.csv')
    grouped.to_csv(out_path)
    print(f"✓ 保存按配置的种子聚合统计: {out_path}")
    return grouped


def save_seed2048_histories(df, histories, output_dir):
    """提取所有 seed=2048 的实验的完整训练历史曲线数据。
    为每个condition（配置组合）保存一个CSV文件。
    """
    seed2048_dir = os.path.join(output_dir, 'seed2048_histories')
    os.makedirs(seed2048_dir, exist_ok=True)

    df_2048 = df[df['seed'] == 2048].copy()
    if df_2048.empty:
        print("⚠️ 没有找到 seed=2048 的实验")
        return

    print(f"\n提取 seed=2048 的训练曲线数据...")
    saved_count = 0

    for idx, row in df_2048.iterrows():
        exp_name = row['exp_name']
        if exp_name not in histories:
            continue

        # 构建condition标识
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
        print(f"  ✓ {condition_name}.csv")

    print(f"✓ 保存了 {saved_count} 个 seed=2048 的训练历史文件到: {seed2048_dir}")


def plot_learning_curves(histories, df, save_dir, top_n=5):
    """绘制学习曲线"""
    print("\n📈 生成学习曲线...")
    
    # 分别为每种模型类型绘制Top实验
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        
        if df_model.empty:
            continue
        
        # 找出Top N实验
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
        print(f"✓ 保存: {filename}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  实验结果提取工具")
    print("  支持: 具身模型 (EC_*) + 单图像模型 (SI_*) + 序列池化模型 (SP_*)")
    print("="*60 + "\n")
    
    # 配置路径
    base_dir = '/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Cognitive_Embodied_Counting'
    experiments_dir = os.path.join(base_dir, 'experiments')
    output_dir = os.path.join(base_dir, 'analysis_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    df, histories = extract_all_experiments(experiments_dir)
    
    if df.empty:
        print("❌ 没有找到任何实验数据！")
        return
    
    # 保存原始数据
    csv_path = os.path.join(output_dir, 'all_experiments_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 保存汇总数据: {csv_path}")
    
    # 分别保存具身、单图像和序列池化的数据
    for model_type in df['model_type'].unique():
        df_model = df[df['model_type'] == model_type]
        # 处理模型类型名称中的下划线
        model_type_clean = model_type.lower().replace(' ', '_')
        csv_model_path = os.path.join(output_dir, f'{model_type_clean}_experiments.csv')
        df_model.to_csv(csv_model_path, index=False)
        print(f"✓ 保存{model_type}数据: {csv_model_path}")
    
    # 生成统计摘要
    summary = generate_summary_statistics(df)
    if summary is not None:
        summary_path = os.path.join(output_dir, 'comparison_statistics.csv')
        summary.to_csv(summary_path)
        print(f"✓ 保存对比统计: {summary_path}")
    
    # 生成同配置不同seed的聚合统计
    save_seed_aggregates(df, output_dir)

    # 提取 seed=2048 的训练历史
    save_seed2048_histories(df, histories, output_dir)
    
    # 打印Top实验
    print("\n" + "="*60)
    print("🏆 最佳实验对比")
    print("="*60)
    
    for model_type in ['Embodied', 'Single_Image', 'Sequence_Pooling']:
        df_model = df[df['model_type'] == model_type]
        if df_model.empty:
            continue
        
        print(f"\n{'='*60}")
        print(f"  {model_type} 模型 - Top 5")
        print(f"{'='*60}")
        
        sort_col = 'best_val_true_final_count_accuracy'
        top5 = df_model.nlargest(5, sort_col)
        
        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n#{rank} {row['exp_name']}")
            print(f"  数据规模: {row.get('data_scale', 'N/A')}%")
            print(f"  预训练: {row.get('use_pretrain', 'N/A')}")
            
            if model_type == 'Embodied':
                print(f"  模态门控: {row.get('use_modality_gate', 'N/A')}")
                print(f"  课程学习: {row.get('curriculum_mode', 'N/A')}")
                print(f"  True Final Accuracy: {row['best_val_true_final_count_accuracy']:.4f}")
                print(f"  Final Accuracy: {row.get('best_val_final_count_accuracy', 0):.4f}")
                print(f"  Count Accuracy: {row.get('best_val_count_accuracy', 0):.4f}")
                print(f"  Joint MSE: {row.get('best_val_joint_mse', 0):.6f}")
            elif model_type == 'Sequence_Pooling':
                print(f"  池化策略: {row.get('pooling_strategy', 'N/A')}")
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (作为Final Count): {row['best_val_true_final_count_accuracy']:.4f}")
            else:  # Single_Image
                print(f"  Accuracy: {row.get('best_val_accuracy', 0):.4f}")
                print(f"  (作为Final Count): {row['best_val_true_final_count_accuracy']:.4f}")
            
            print(f"  最佳Epoch: {row['best_epoch']}")
            print(f"  Validation Loss: {row['best_val_loss']:.4f}")
    
    print("\n" + "="*60)
    print(f"✅ 所有结果已保存到: {output_dir}")
    print("="*60)
    print("\n生成的文件:")
    print(f"  📄 all_experiments_summary.csv - 所有实验的详细指标")
    print(f"  📄 embodied_experiments.csv - 具身模型实验")
    print(f"  📄 single_image_experiments.csv - 单图像模型实验")
    print(f"  📄 sequence_pooling_experiments.csv - 序列池化模型实验")
    print(f"  📄 comparison_statistics.csv - 模型对比统计")
    print(f"  📄 by_config_seed_stats.csv - 按配置的种子聚合统计")
    print(f"  📄 seed2048_histories/ - seed=2048的训练曲线数据")


if __name__ == '__main__':
    main()
