import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import wandb

# Make sure we can import local modules
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, 'Models'))
sys.path.append(os.path.join(CUR_DIR, 'Data_loader'))

from Single_Image_Classifier import create_single_image_model  # type: ignore
from DataLoader_single_image import get_single_image_data_loaders  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SingleImageTrainer:
    """Trainer for single image classification using wandb logging."""

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 config: dict,
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

        # Criterion
        self.ce_criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)  # [B, C, H, W]
            labels = batch['label'].to(self.device).long()  # [B]

            self.optimizer.zero_grad()
            logits = self.model(images)  # [B, num_classes]
            loss = self.ce_criterion(logits, labels)
            loss.backward()
            
            if self.grad_clip is not None and self.grad_clip > 0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            epoch_loss += loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.3f}",
            })

        epoch_loss /= max(1, len(self.train_loader))
        epoch_acc = correct / max(1, total)

        # log to wandb
        wandb.log({
            'train/loss': epoch_loss,
            'train/accuracy': epoch_acc,
            'lr': self.optimizer.param_groups[0]['lr']
        }, step=epoch)

        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Collections for confusion matrix
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"Val {epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).long()

            logits = self.model(images)
            loss = self.ce_criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            epoch_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.3f}",
            })

        epoch_loss /= max(1, len(self.val_loader))
        epoch_acc = correct / max(1, total)

        # Per-class accuracy
        per_class_acc = {}
        unique_labels = sorted(set(all_labels))
        for label in unique_labels:
            mask = np.array(all_labels) == label
            if mask.sum() > 0:
                class_correct = ((np.array(all_preds)[mask]) == label).sum()
                per_class_acc[f'val/class_{label}_accuracy'] = class_correct / mask.sum()

        # log to wandb
        wandb.log({
            'val/loss': epoch_loss,
            'val/accuracy': epoch_acc,
            **per_class_acc
        }, step=epoch)

        # Save confusion matrix every 10 epochs
        if epoch % 10 == 0:
            cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(unique_labels)))
            ax.set_yticks(range(len(unique_labels)))
            ax.set_xticklabels(unique_labels)
            ax.set_yticklabels(unique_labels)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Epoch {epoch}')
            
            cm_path = os.path.join(self.save_dir, f'confusion_matrix_epoch_{epoch}.png')
            plt.savefig(cm_path, bbox_inches='tight', dpi=150)
            plt.close()
            wandb.log({"val/confusion_matrix": wandb.Image(cm_path)}, step=epoch)

        # Save best model
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            self._save_checkpoint(epoch, best=True)

        # Periodic checkpoint
        if epoch % self.config.get('save_every', 10) == 0:
            self._save_checkpoint(epoch, best=False)

        return epoch_loss, epoch_acc

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
                       num_workers: int):
    train_loader, val_loader = get_single_image_data_loaders(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        data_root=data_root,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=num_workers,
        normalize_images=True,
        custom_image_norm_stats=None
    )
    return train_loader, val_loader


def build_model(use_pretrain: bool, num_classes: int, device: torch.device):
    model = create_single_image_model(
        num_classes=num_classes,
        use_pretrain=use_pretrain,
        input_channels=3
    )
    return model.to(device)


def run_training_once(exp_name: str,
                      data_root: str,
                      train_csv: str,
                      val_csv: str,
                      total_epochs: int = 250,
                      batch_size: int = 32,
                      sequence_length: int = 11,
                      num_workers: int = 4,
                      learning_rate: float = 1e-4,
                      use_pretrain: bool = True,
                      seed: int = 42,
                      project: str = 'single-image-counting',
                      save_dir: str | None = None):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save dir (create before wandb.init)
    if save_dir is None or len(str(save_dir).strip()) == 0:
        save_dir = os.path.join(CUR_DIR, 'experiments/single_image_250', exp_name)
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
            'use_pretrain': use_pretrain,
            'seed': seed,
        }
    )

    # data
    train_loader, val_loader = build_data_loaders(
        data_root=data_root,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=num_workers
    )

    # model
    model = build_model(use_pretrain, num_classes=11, device=device)

    # trainer
    trainer = SingleImageTrainer(
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
            'save_every': 2,
        },
        save_dir=save_dir,
    )

    history = []
    for epoch in range(total_epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.validate(epoch)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        })

    # save history json
    hist_path = os.path.join(save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    wandb.save(hist_path)

    wandb.finish()

    return history
