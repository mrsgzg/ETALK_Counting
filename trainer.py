import os
import sys
import time
import json
import math
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# plotting and metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# logging
import wandb

# Make sure we can import local modules
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, 'Models'))
sys.path.append(os.path.join(CUR_DIR, 'Data_loader'))

from Embody_Counting_Model import create_model  # type: ignore
from Data_loader_embodiment import get_ball_counting_data_loaders  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbodiedTrainer:
    """Trainer for simplified embodied counting model using wandb logging."""

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 config: Dict[str, Any],
                 save_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.save_dir = save_dir
        self.ckpt_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=config.get('adam_betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 1e-5),
        )

        sched_type = config.get('scheduler_type', 'cosine')
        if sched_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.get('total_epochs', 100)
            )
        elif sched_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, config.get('step_size', 30)), gamma=0.1
            )
        else:
            self.scheduler = None

        self.grad_clip = config.get('grad_clip_norm', 1.0)

        # Loss weights
        self.count_loss_weight = config.get('count_loss_weight', 1.0)
        self.joint_loss_weight = config.get('joint_loss_weight', 0.3)

        # Criterion
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()

        self.best_val_loss = float('inf')

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], sequence_data: Dict[str, torch.Tensor]):
        # Counting: per-frame cross-entropy
        logits = outputs['counts']  # [B, S, C]
        labels = sequence_data['labels'].to(self.device).long()  # [B, S]
        B, S, C = logits.shape
        count_loss = self.ce_criterion(logits.view(B * S, C), labels.view(B * S))

        # Joints: predict next-step joints => compare pred[t] to GT[t+1]
        pred_joints = outputs['joints']              # [B, S, 2]
        gt_joints = sequence_data['joints'].to(self.device)  # [B, S, 2]
        if S > 1:
            pred_next = pred_joints[:, :-1]  # [B, S-1, 2]
            gt_next = gt_joints[:, 1:]       # [B, S-1, 2]
            joint_loss = self.mse_criterion(pred_next, gt_next)
        else:
            # Degenerate case, compare same step
            joint_loss = self.mse_criterion(pred_joints, gt_joints)

        total_loss = self.count_loss_weight * count_loss + self.joint_loss_weight * joint_loss
        return total_loss, count_loss, joint_loss

    @torch.no_grad()
    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], sequence_data: Dict[str, torch.Tensor]):
        logits = outputs['counts']  # [B, S, C]
        labels = sequence_data['labels'].to(self.device).long()  # [B, S]
        preds = torch.argmax(logits, dim=-1)  # [B, S]

        # 1. overall accuracy (all timesteps) - 所有时间步的平均准确率
        count_acc = (preds == labels).float().mean().item()

        # 2. final count accuracy (last timestep) - 序列最后一个时间步的准确率
        final_pred = preds[:, -1]
        final_target = labels[:, -1]
        final_count_acc = (final_pred == final_target).float().mean().item()

        # 3. true final count accuracy (based on actual sequence length) - 根据真实序列长度的最终准确率
        batch_size = preds.shape[0]
        true_final_correct = 0
        for i in range(batch_size):
            # 找到真实的最终位置（最大标签值的位置）
            max_label = labels[i].max()
            final_positions = (labels[i] == max_label).nonzero(as_tuple=True)[0]
            if len(final_positions) > 0:
                true_final_pos = final_positions[0].item()
                if preds[i, true_final_pos] == labels[i, true_final_pos]:
                    true_final_correct += 1
        true_final_count_acc = true_final_correct / batch_size

        # Joint MSE on next timestep
        pred_joints = outputs['joints']
        gt_joints = sequence_data['joints'].to(self.device)
        S = pred_joints.shape[1]
        if S > 1:
            joint_mse = torch.mean((pred_joints[:, :-1] - gt_joints[:, 1:]) ** 2).item()
        else:
            joint_mse = torch.mean((pred_joints - gt_joints) ** 2).item()

        return {
            'count_accuracy': count_acc,
            'final_count_accuracy': final_count_acc,
            'true_final_count_accuracy': true_final_count_acc,
            'joint_mse': joint_mse,
        }

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0
        epoch_count_loss = 0.0
        epoch_joint_loss = 0.0
        epoch_metrics = {
            'count_accuracy': 0.0,
            'final_count_accuracy': 0.0,
            'true_final_count_accuracy': 0.0,
            'joint_mse': 0.0,
        }
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for batch in pbar:
            sequence_data = batch['sequence_data']
            # move tensors to device
            sequence_data = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in sequence_data.items()
            }

            self.optimizer.zero_grad()
            outputs = self.model(sequence_data)
            total_loss, count_loss, joint_loss = self._compute_losses(outputs, sequence_data)
            total_loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            metrics = self._compute_metrics(outputs, sequence_data)

            # accumulate
            epoch_loss += total_loss.item()
            epoch_count_loss += count_loss.item()
            epoch_joint_loss += joint_loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            steps += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{metrics['count_accuracy']:.3f}",
            })

        # averages
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, steps)
        epoch_loss /= max(1, steps)
        epoch_count_loss /= max(1, steps)
        epoch_joint_loss /= max(1, steps)

        # log to wandb
        wandb.log({
            'train/total_loss': epoch_loss,
            'train/count_loss': epoch_count_loss,
            'train/joint_loss': epoch_joint_loss,
            'train/count_accuracy': epoch_metrics['count_accuracy'],
            'train/final_count_accuracy': epoch_metrics['final_count_accuracy'],
            'train/true_final_count_accuracy': epoch_metrics['true_final_count_accuracy'],
            'train/joint_mse': epoch_metrics['joint_mse'],
            'lr': self.optimizer.param_groups[0]['lr']
        }, step=epoch)

        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        epoch_loss = 0.0
        epoch_count_loss = 0.0
        epoch_joint_loss = 0.0
        epoch_metrics = {
            'count_accuracy': 0.0,
            'final_count_accuracy': 0.0,
            'true_final_count_accuracy': 0.0,
            'joint_mse': 0.0,
        }
        steps = 0

        # Collections for confusion matrix
        all_final_preds = []
        all_final_labels = []

        pbar = tqdm(self.val_loader, desc=f"Val   {epoch}")
        for batch in pbar:
            sequence_data = batch['sequence_data']
            sequence_data = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in sequence_data.items()
            }

            outputs = self.model(sequence_data, return_hidden_states=False)
            total_loss, count_loss, joint_loss = self._compute_losses(outputs, sequence_data)

            metrics = self._compute_metrics(outputs, sequence_data)

            epoch_loss += total_loss.item()
            epoch_count_loss += count_loss.item()
            epoch_joint_loss += joint_loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            steps += 1

            # final step preds/labels for confusion matrix
            preds = torch.argmax(outputs['counts'], dim=-1)  # [B,S]
            all_final_preds.append(preds[:, -1].detach().cpu())
            all_final_labels.append(sequence_data['labels'][:, -1].detach().cpu())

        # averages
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, steps)
        epoch_loss /= max(1, steps)
        epoch_count_loss /= max(1, steps)
        epoch_joint_loss /= max(1, steps)

        # confusion matrix figure
        all_final_preds = torch.cat(all_final_preds, dim=0).numpy()
        all_final_labels = torch.cat(all_final_labels, dim=0).numpy()
        cm = confusion_matrix(all_final_labels, all_final_preds, labels=list(range(self.config.get('num_classes', 11))))

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(self.config.get('num_classes', 11))),
                    yticklabels=list(range(self.config.get('num_classes', 11))))
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True')
        plt.xlabel('Pred')
        wandb.log({'val/confusion_matrix': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

        # log to wandb
        wandb.log({
            'val/total_loss': epoch_loss,
            'val/count_loss': epoch_count_loss,
            'val/joint_loss': epoch_joint_loss,
            'val/count_accuracy': epoch_metrics['count_accuracy'],
            'val/final_count_accuracy': epoch_metrics['final_count_accuracy'],
            'val/true_final_count_accuracy': epoch_metrics['true_final_count_accuracy'],
            'val/joint_mse': epoch_metrics['joint_mse'],
        }, step=epoch)

        # track best by lowest validation loss
        is_best = epoch_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = epoch_loss
            self._save_checkpoint(epoch, best=True)
        # also save periodic checkpoints
        save_every = self.config.get('save_every', 10)
        if save_every and (epoch + 1) % save_every == 0:
            self._save_checkpoint(epoch, best=False)

        return epoch_loss, epoch_metrics

    def _save_checkpoint(self, epoch: int, best: bool = False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        fname = 'best.pt' if best else f'epoch_{epoch+1}.pt'
        path = os.path.join(self.ckpt_dir, fname)
        torch.save(state, path)
        wandb.save(path)


def build_data_loaders(data_root: str,
                       train_csv: str,
                       val_csv: str,
                       batch_size: int,
                       sequence_length: int,
                       num_workers: int,
                       curriculum_mode: str = 'random',
                       shuffle_joints: bool = False):
    train_loader, val_loader = get_ball_counting_data_loaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        data_root=data_root,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=num_workers,
        normalize_images=True,
        custom_image_norm_stats=None,
        shuffle_joints=shuffle_joints,
        curriculum_mode=curriculum_mode,
        seed=42,
    )
    return train_loader, val_loader


def build_model(model_config: Dict[str, Any], use_pretrain: bool, use_modality_gate: bool, device: torch.device):
    cfg = {
        'image_mode': 'rgb',
        'model_config': model_config.copy(),
    }
    model = create_model(cfg, use_pretrain=use_pretrain, use_modality_gate=use_modality_gate)
    return model.to(device)


def run_training_once(exp_name: str,
                      data_root: str,
                      train_csv: str,
                      val_csv: str,
                      model_config: Dict[str, Any],
                      total_epochs: int = 50,
                      batch_size: int = 16,
                      sequence_length: int = 11,
                      num_workers: int = 2,
                      learning_rate: float = 1e-4,
                      curriculum_mode: str = 'random',
                      shuffle_joints: bool = False,
                      use_pretrain: bool = True,
                      use_modality_gate: bool = True,
                      seed: int = 42,
                      project: str = 'embodied-counting',
                      save_dir: str | None = None):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save dir (create before wandb.init)
    if save_dir is None or len(str(save_dir).strip()) == 0:
        save_dir = os.path.join(CUR_DIR, 'experiments', exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # wandb init - 离线模式，使用实验目录
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        project=project, 
        name=exp_name,
        dir=save_dir,  # wandb 文件存储在实验目录下
        config={
            'total_epochs': total_epochs,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'learning_rate': learning_rate,
            'curriculum_mode': curriculum_mode,
            'shuffle_joints': shuffle_joints,
            'use_pretrain': use_pretrain,
            'use_modality_gate': use_modality_gate,
            'model_config': model_config,
        }
    )

    # data
    train_loader, val_loader = build_data_loaders(
        data_root=data_root,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=num_workers,
        curriculum_mode=curriculum_mode,
        shuffle_joints=shuffle_joints,
    )

    # model
    model = build_model(model_config, use_pretrain, use_modality_gate, device)

    # trainer
    trainer = EmbodiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config={
            'learning_rate': learning_rate,
            'adam_betas': (0.9, 0.999),
            'weight_decay': 1e-5,
            'scheduler_type': 'cosine',
            'total_epochs': total_epochs,
            'grad_clip_norm': 1.0,
            'count_loss_weight': 1.0,
            'joint_loss_weight': 0.3,
            'save_every': 2,
            'num_classes': 11,
        },
        save_dir=save_dir,
    )

    history = []
    for epoch in range(total_epochs):
        train_loss, train_metrics = trainer.train_epoch(epoch)
        val_loss, val_metrics = trainer.validate(epoch)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

    # save history json
    hist_path = os.path.join(save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    wandb.save(hist_path)

    wandb.finish()

    return history
