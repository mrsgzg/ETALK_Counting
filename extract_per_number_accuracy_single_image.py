"""
Extract per-number accuracy across training epochs for single-image models.
Reads checkpoints and computes per-class accuracy for each epoch,
filtered by train_pct and pre_type. Outputs CSV files per model.
"""

import os
import sys
import argparse
import glob
import json
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import confusion_matrix

CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, 'Models'))
sys.path.append(os.path.join(CUR_DIR, 'Data_loader'))

from Single_Image_Classifier import create_single_image_model
from DataLoader_single_image import get_single_image_data_loaders


def get_device(device_arg: str) -> torch.device:
    """
    Resolve the compute device from a string argument.
    
    Args:
        device_arg: 'auto', 'cpu', 'cuda', or a specific device like 'cuda:0'
    
    Returns:
        torch.device
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA is available — using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("CUDA not available — using CPU")
    elif device_arg.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(device_arg)
            gpu_id = int(device_arg.split(':')[1]) if ':' in device_arg else 0
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print("CUDA requested but not available — falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple:
    """
    Load a model checkpoint and reconstruct the SingleImageClassifier.
    
    Args:
        checkpoint_path: path to the checkpoint file
        device: compute device to load onto
    
    Returns:
        model: loaded model in eval mode
        config: config dict extracted from the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available
    config = {}
    if isinstance(ckpt, dict) and 'config' in ckpt:
        config = ckpt['config']
    else:
        config = {'num_classes': 11, 'use_pretrain': False}
    
    num_classes = config.get('num_classes', 11)
    
    # Instantiate model (weights come from checkpoint, not pretrained)
    model = create_single_image_model(
        num_classes=num_classes,
        use_pretrain=False,
        input_channels=3
    )
    
    # Load state dict — try several common key names
    if isinstance(ckpt, dict):
        for key in ['model_state_dict', 'model_state', 'state_dict']:
            if key in ckpt:
                model.load_state_dict(ckpt[key], strict=False)
                break
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def load_state_dict_only(checkpoint_path: str, device: torch.device) -> Dict:
    """Load only the state dict from a checkpoint without creating a new model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = None
    
    if isinstance(ckpt, dict):
        for key in ['model_state_dict', 'model_state', 'state_dict']:
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            possible = {k: v for k, v in ckpt.items() if k not in ['config']}
            state = possible if possible else None
    
    if state is None:
        state = ckpt
    
    return state


@torch.no_grad()
def evaluate_per_number_accuracy(model, dataloader, device, num_classes: int = 11) -> Dict:
    """
    Evaluate per-class accuracy over the entire dataloader.
    
    Args:
        model: single-image classifier in eval mode
        dataloader: DataLoader to iterate over
        device: compute device
        num_classes: number of output classes
    
    Returns:
        dict mapping class index to accuracy on that class
    """
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Forward pass through encoder and classifier
            features = model.visual_encoder(images)
            logits = model.classifier(features)
            predictions = logits.argmax(dim=1)
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            # Free GPU tensors to keep memory usage low
            if device.type == 'cuda':
                del images, labels, features, logits, predictions
    
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    # Compute per-class accuracy
    per_class_accuracy = {}
    for class_idx in range(num_classes):
        mask = (all_labels == class_idx)
        if mask.sum() > 0:
            accuracy = (all_predictions[mask] == class_idx).mean()
            per_class_accuracy[class_idx] = float(accuracy)
        else:
            per_class_accuracy[class_idx] = 0.0
    
    return per_class_accuracy


def process_single_model(args_tuple):
    """Wrapper function for multiprocessing — unpacks arguments and calls the main extractor."""
    exp_dir, data_root, train_csv, val_csv, device_str, output_dir, batch_size, num_data_workers = args_tuple
    device = torch.device(device_str)
    return extract_accuracy_for_model(exp_dir, data_root, train_csv, val_csv, device, 
                                      output_dir, batch_size, num_data_workers)


def extract_accuracy_for_model(exp_dir: str,
                                data_root: str,
                                train_csv: str,
                                val_csv: str,
                                device: torch.device,
                                output_dir: str,
                                batch_size: int = 64,
                                num_data_workers: int = 4) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Extract per-number accuracy for all checkpoints in an experiment directory.
    
    Returns:
        (model_name, epoch_dataframe, summary_dataframe)
    """
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return None, None, None
    
    # Get all epoch checkpoints sorted by filename
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    if not checkpoints:
        print(f"No epoch checkpoints found in {ckpt_dir}")
        return None, None, None
    
    model_name = os.path.basename(exp_dir)
    
    print(f"\n{'='*70}")
    print(f"Processing: {model_name}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"{'='*70}")
    
    # Load validation data once (reused across all checkpoints)
    t_start_data = time.time()
    try:
        _, val_loader = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_data_workers,
            sequence_length=11,
            normalize_images=True
        )
        t_data = time.time() - t_start_data
        print(f"Validation data loaded in {t_data:.2f}s")
        print(f"  Batch size: {batch_size}, Workers: {num_data_workers}")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None, None, None
    
    # Initialize model ONCE and reuse across all checkpoints
    print(f"Initializing model template...")
    t_start_model = time.time()
    try:
        model, cfg = load_checkpoint(checkpoints[0], device)
        model.eval()
        
        # Enable cudnn benchmark for faster CUDA inference
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"CUDA optimizations enabled (cudnn.benchmark=True)")
        t_model = time.time() - t_start_model
        print(f"Model initialized in {t_model:.2f}s")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None, None
    
    # Evaluate each checkpoint
    epoch_results = []
    load_times = []
    eval_times = []
    
    pbar = tqdm(checkpoints, desc=f"Evaluating {model_name}")
    for ckpt_path in pbar:
        try:
            # Extract epoch number from filename
            epoch_num = int(os.path.basename(ckpt_path).split('_')[1].split('.')[0])
            
            # Load only the state dict into the existing model (avoids repeated allocation)
            t_load_start = time.time()
            state = load_state_dict_only(ckpt_path, device)
            model.load_state_dict(state, strict=False)
            t_load = time.time() - t_load_start
            load_times.append(t_load)
            
            # Clear GPU cache between checkpoints
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Evaluate
            t_eval_start = time.time()
            per_number_acc = evaluate_per_number_accuracy(model, val_loader, device, num_classes=11)
            t_eval = time.time() - t_eval_start
            eval_times.append(t_eval)
            
            # Update progress bar with timing info
            postfix_dict = {
                'load': f'{t_load:.2f}s',
                'eval': f'{t_eval:.2f}s',
                'total': f'{t_load+t_eval:.2f}s',
                'avg': f'{np.mean(load_times)+np.mean(eval_times):.2f}s'
            }
            
            # Add GPU memory usage to progress bar if on CUDA
            if device.type == 'cuda':
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**3
                postfix_dict['gpu_mem'] = f'{gpu_mem:.2f}GB'
            
            pbar.set_postfix(postfix_dict)
            
            # Store per-class accuracies for this epoch
            row = {'epoch': epoch_num, 'model': model_name}
            for num in range(11):
                row[f'accuracy_number_{num}'] = per_number_acc.get(num, 0.0)
            epoch_results.append(row)
            
            # Clear GPU cache after evaluation
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {ckpt_path}: {e}")
            continue
    
    if not epoch_results:
        print(f"No results extracted for {model_name}")
        return None, None, None
    
    # Build epoch-level dataframe
    epoch_df = pd.DataFrame(epoch_results)
    epoch_df = epoch_df.sort_values('epoch').reset_index(drop=True)
    
    # Build summary statistics dataframe
    summary_data = []
    for num in range(11):
        col_name = f'accuracy_number_{num}'
        if col_name in epoch_df.columns:
            accuracies = epoch_df[col_name].values
            summary_data.append({
                'number': num,
                'model': model_name,
                'mean_accuracy': accuracies.mean(),
                'std_accuracy': accuracies.std(),
                'min_accuracy': accuracies.min(),
                'max_accuracy': accuracies.max(),
                'final_accuracy': accuracies[-1] if len(accuracies) > 0 else 0.0,
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save individual model results
    os.makedirs(output_dir, exist_ok=True)
    epoch_csv = os.path.join(output_dir, f'per_number_accuracy_by_epoch_{model_name}.csv')
    summary_csv = os.path.join(output_dir, f'per_number_accuracy_summary_{model_name}.csv')
    
    epoch_df.to_csv(epoch_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"Saved epoch results: {epoch_csv}")
    print(f"Saved summary results: {summary_csv}")
    
    # Print timing summary
    if load_times:
        print(f"\nTiming summary:")
        print(f"  - Mean checkpoint load time: {np.mean(load_times):.2f}s")
        print(f"  - Mean evaluation time: {np.mean(eval_times):.2f}s")
        print(f"  - Mean time per checkpoint: {np.mean(load_times) + np.mean(eval_times):.2f}s")
        print(f"  - Total load time: {sum(load_times):.2f}s ({sum(load_times)/60:.1f} min)")
        print(f"  - Total eval time: {sum(eval_times):.2f}s ({sum(eval_times)/60:.1f} min)")
        
        if device.type == 'cuda':
            print(f"  - Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    # Free GPU memory after processing this model
    if device.type == 'cuda':
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    return model_name, epoch_df, summary_df


def main():
    parser = argparse.ArgumentParser(description='Extract per-number accuracy across epochs for single image models')
    parser.add_argument('--data_root', type=str,
                        default='data/ball_data_collection')
    parser.add_argument('--train_csv', type=str,
                        default='data/Tools_script/ball_counting_dataset_train_10.csv')
    parser.add_argument('--val_csv', type=str,
                        default='data/Tools_script/ball_counting_dataset_val.csv')
    parser.add_argument('--experiments_dir', type=str,
                        help='Directory containing experiment folders with checkpoints')
    parser.add_argument('--output_dir', type=str, default='per_number_accuracy_results_si',
                        help='Output directory for CSV files')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                        help='Device to use for computation (default: auto — use GPU if available)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (larger values recommended for GPU, default: 64)')
    parser.add_argument('--num_data_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel model workers (default: 1; set >1 for multi-GPU)')
    parser.add_argument('--train_pct', type=str, default='100pct',
                        choices=['100pct', '50pct', '10pct', 'all'],
                        help='Filter experiments by training percentage; use "all" to include all')
    parser.add_argument('--pre_type', type=str, default='all',
                        choices=['pre0', 'pre1', 'all'],
                        help='Filter by pretrained type (pre0=random init, pre1=pretrained); use "all" to include all')
    
    args = parser.parse_args()
    
    # Resolve compute device
    device = get_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve experiments directory
    if args.experiments_dir is None:
        experiments_dir = os.path.join(CUR_DIR, 'experiments')
    else:
        experiments_dir = args.experiments_dir
    
    if not os.path.exists(experiments_dir):
        print(f"Error: experiments directory not found: {experiments_dir}")
        return
    
    # Collect all experiment directories
    all_exp_dirs = sorted([d for d in glob.glob(os.path.join(experiments_dir, '*'))
                           if os.path.isdir(d)])
    
    # Filter: must start with SI_ and match train_pct / pre_type filters
    exp_dirs = []
    for d in all_exp_dirs:
        name = os.path.basename(d)
        if not name.startswith('SI_'):
            continue
        if args.train_pct != 'all' and args.train_pct not in name:
            continue
        if args.pre_type != 'all' and args.pre_type not in name:
            continue
        exp_dirs.append(d)
    
    if not exp_dirs:
        print(f"No experiment directories matched filters in {experiments_dir}")
        print(f"  Prefix required: SI_")
        print(f"  Training percentage filter: {args.train_pct}")
        print(f"  Pretrained type filter: {args.pre_type}")
        print(f"Total experiments found: {len(all_exp_dirs)}")
        return
    
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"{'='*70}")
    print(f"Found {len(exp_dirs)} experiment directories after filtering (total: {len(all_exp_dirs)})")
    print(f"Filters -> prefix: SI_, train_pct: {args.train_pct}, pre_type: {args.pre_type}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data workers: {args.num_data_workers}")
    print(f"Model workers: {args.num_workers}")
    print(f"{'='*70}\n")
    
    # Accumulators for results across all models
    all_epoch_dfs = []
    all_summary_dfs = []
    
    # Build argument tuples for each model
    model_args = [
        (exp_dir, args.data_root, args.train_csv, args.val_csv, str(device), 
         args.output_dir, args.batch_size, args.num_data_workers)
        for exp_dir in exp_dirs
    ]
    
    # Run in parallel or sequentially
    if args.num_workers > 1 and len(exp_dirs) > 1:
        print(f"{'='*70}")
        print(f"Running {len(exp_dirs)} models with {args.num_workers} parallel workers")
        print(f"Note: for GPU usage, typically keep num_workers=1 to avoid device conflicts")
        print(f"{'='*70}\n")
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(process_single_model, model_args)
    else:
        print(f"{'='*70}")
        print(f"Running {len(exp_dirs)} models sequentially")
        print(f"{'='*70}\n")
        results = [process_single_model(args) for args in model_args]
    
    # Collect valid results
    for model_name, epoch_df, summary_df in results:
        if epoch_df is not None:
            all_epoch_dfs.append(epoch_df)
            all_summary_dfs.append(summary_df)
    
    if all_epoch_dfs:
        # Combine all models into single files
        combined_epoch_df = pd.concat(all_epoch_dfs, ignore_index=True)
        combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
        
        # Save combined CSV files
        combined_epoch_csv = os.path.join(args.output_dir, 'per_number_accuracy_all_models_SI.csv')
        combined_summary_csv = os.path.join(args.output_dir, 'per_number_accuracy_summary_all_models_SI.csv')
        
        combined_epoch_df.to_csv(combined_epoch_csv, index=False)
        combined_summary_df.to_csv(combined_summary_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"Extraction complete!")
        print(f"{'='*70}")
        print(f"\nProcessed {len(all_epoch_dfs)} models")
        print(f"Total epoch rows: {len(combined_epoch_df)}")
        print(f"\nOutput files:")
        print(f"  Combined epoch results: {combined_epoch_csv}")
        print(f"  Combined summary results: {combined_summary_csv}")
        print(f"  Individual model files saved to: {args.output_dir}")
        
        if device.type == 'cuda':
            print(f"\nGPU memory usage:")
            print(f"  - Peak: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"  - Current: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print()
    else:
        print(f"\n{'='*70}")
        print("No valid results extracted from any model.")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()