"""
Embodied model analysis script: visual head Grad-CAM, LSTM joint PCA, gates, softmax/confusion.

Outputs (under --output_dir/embodied_analysis_YYYYMMDD_HHMMSS):
- features/lstm_pca_by_timestep.png, lstm_pca_by_label.png, optional trajectories
- attention_maps/gradcam_samples.png (CAM + per-sample softmax bars)
- predictions/confusion_matrix(_normalized).png, per_class_accuracy.png
- predictions/softmax_over_time.png (optional), softmax_summary.png
- gates/gate_weights_over_time.png (if gate enabled)
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, 'Models'))
sys.path.append(os.path.join(CUR_DIR, 'Data_loader'))

from Models.Embody_Counting_Model import SimplifiedEmbodiedCountingModel
from Data_loader.Data_loader_embodiment import get_ball_counting_data_loaders
from visualization_utils import (
    plot_confusion_matrix,
    generate_gradcam_overlay,
)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_embodied_checkpoint(checkpoint_path: str, device: torch.device, model_config_overrides: dict = None):
    print("\n" + "="*60)
    print(f"加载具身模型: {checkpoint_path}")
    print("="*60)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to get config
    cfg = {}
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg.update(ckpt['config'])
    if model_config_overrides:
        cfg.update(model_config_overrides)

    # Defaults aligned with training
    defaults = dict(
        use_pretrain=True,
        lstm_layers=2,
        lstm_hidden_size=512,
        feature_dim=256,
        joint_dim=2,
        input_channels=3,
        dropout=0.1,
        num_classes=11,
        use_modality_gate=True,
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    model = SimplifiedEmbodiedCountingModel(**cfg)

    # Robust state_dict loading
    state = None
    for key in ['model_state_dict', 'model_state', 'state_dict']:
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if state is None and isinstance(ckpt, dict):
        # filter
        possible = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        state = possible if possible else None
    if state is None:
        # maybe ckpt itself is state_dict
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠ Unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    print("✓ 模型加载完成\n")
    return model, cfg


def _save_pca_csv(path: str, sample_ids: np.ndarray, timesteps: np.ndarray, labels: np.ndarray, comps: np.ndarray, header_cols: list[str]):
    """Save PCA/TSNE components with metadata to CSV.
    comps: [N*T, k]
    header_cols: list of column names, length = comps.shape[1] + 3 (sample_id, timestep, label + comps)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.column_stack([sample_ids, timesteps, labels, comps])
    header = ",".join(header_cols)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")


def extract_sequence_outputs(model: SimplifiedEmbodiedCountingModel, dataloader: DataLoader, device: torch.device):
    print("\n" + "="*60)
    print("提取序列预测、隐藏状态与门控权重…")
    print("="*60)

    all_logits = []  # [N, T, C]
    all_labels = []  # [N]
    all_hidden = []  # [N, T, H] - layer 1 (top)
    all_hidden_layer2 = []  # [N, T, H] - layer 2 (bottom)
    all_gates = []   # [N, T, 2]

    # Setup hooks to capture LSTM layer outputs
    lstm_layer_outputs = {'layer1': [], 'layer2': []}
    
    def hook_lstm(layer_idx):
        def _hook(module, input, output):
            # output is (output, (h_n, c_n))
            h_n = output[1][0]  # [num_layers, B, hidden_size]
            # Capture both layers
            if h_n.shape[0] >= 2:
                lstm_layer_outputs['layer1'].append(h_n[-1].detach().clone())  # top layer
                lstm_layer_outputs['layer2'].append(h_n[-2].detach().clone())  # bottom layer
            elif h_n.shape[0] == 1:
                lstm_layer_outputs['layer1'].append(h_n[0].detach().clone())
        return _hook
    
    lstm_module = model.lstm
    handle = lstm_module.register_forward_hook(hook_lstm(0))

    with torch.no_grad():
        for batch in dataloader:
            lstm_layer_outputs['layer1'].clear()
            lstm_layer_outputs['layer2'].clear()
            
            seq = batch['sequence_data']
            labels = batch['label']

            seq_dev = {
                'images': seq['images'].to(device),
                'joints': seq['joints'].to(device),
            }

            outputs = model(seq_dev, return_hidden_states=True)
            counts = outputs['counts']  # [B, T, C]

            all_logits.append(counts.cpu())
            all_labels.append(labels.long().cpu())

            # Collect LSTM hidden states from hooks
            if lstm_layer_outputs['layer1']:
                layer1_states = torch.stack(lstm_layer_outputs['layer1'])  # [T, B, H]
                all_hidden.append(layer1_states.cpu().transpose(0, 1))  # [B, T, H]
            
            if lstm_layer_outputs['layer2']:
                layer2_states = torch.stack(lstm_layer_outputs['layer2'])  # [T, B, H]
                all_hidden_layer2.append(layer2_states.cpu().transpose(0, 1))  # [B, T, H]
            
            if 'gate_weights' in outputs:
                all_gates.append(outputs['gate_weights'].cpu())

    handle.remove()

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    hidden_layer1 = torch.cat(all_hidden, dim=0).numpy() if all_hidden else None
    hidden_layer2 = torch.cat(all_hidden_layer2, dim=0).numpy() if all_hidden_layer2 else None
    gates = torch.cat(all_gates, dim=0).numpy() if all_gates else None

    # Sequence-level prediction from last timestep
    last_logits = logits[:, -1, :]  # [N, C]
    preds = last_logits.argmax(axis=1)
    acc = (preds == labels).mean() if labels is not None else float('nan')
    print(f"✓ 提取完成: 样本数={logits.shape[0]}, 序列长={logits.shape[1]}, 类别数={logits.shape[2]} | 序列级准确率={acc:.2%}")
    return logits, labels, preds, hidden_layer1, hidden_layer2, gates


def extract_visual_features(model: SimplifiedEmbodiedCountingModel, dataloader: DataLoader, device: torch.device):
    """Extract per-frame visual encoder features for the entire dataset.
    Returns:
      feats_btD: np.ndarray [N, T, D] where D is `model.visual_encoder.feature_dim` (default 256)
      labels:   np.ndarray [N]
    """
    all_feats = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            seq = batch['sequence_data']
            labels = batch['label']
            images = seq['images'].to(device)  # [B,T,C,H,W]
            B, T = images.shape[:2]
            flat_imgs = images.reshape(B*T, *images.shape[2:])
            feats = model.visual_encoder(flat_imgs)  # [B*T, D]
            feats_btD = feats.reshape(B, T, -1).cpu()
            all_feats.append(feats_btD)
            all_labels.append(labels.long().cpu())

    feats_btD = torch.cat(all_feats, dim=0).numpy() if all_feats else None
    labels_np = torch.cat(all_labels, dim=0).numpy() if all_labels else None
    return feats_btD, labels_np


def plot_visual_pca_tsne(visual_btD: np.ndarray, labels: np.ndarray, save_dir: str):
    """Plot PCA (2D+3D) and t-SNE (2D) for visual encoder features.
    - Colors by timestep and by sequence label.
    Outputs PNGs into `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)
    N, T, D = visual_btD.shape
    feats = visual_btD.reshape(N*T, D)
    timesteps = np.repeat(np.arange(T), N)
    labels_rep = np.repeat(labels, T)
    sample_ids = np.repeat(np.arange(N), T)

    # 2D PCA
    pca2d = PCA(n_components=2)
    comps2d = pca2d.fit_transform(feats)
    var2d = pca2d.explained_variance_ratio_

    plt.figure(figsize=(8,6))
    sc = plt.scatter(comps2d[:,0], comps2d[:,1], c=timesteps, cmap='viridis', s=6)
    plt.colorbar(sc, label='timestep (0-based)')
    plt.title(f'Visual Head PCA 2D by timestep (PC1 {var2d[0]*100:.1f}%, PC2 {var2d[1]*100:.1f}%)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'visual_pca_2d_by_timestep.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    sc = plt.scatter(comps2d[:,0], comps2d[:,1], c=labels_rep, cmap='tab10', s=6)
    plt.colorbar(sc, label='ball count')
    plt.title('Visual Head PCA 2D by label (sequence label)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'visual_pca_2d_by_label.png'), dpi=200)
    plt.close()

    # 3D PCA
    pca3d = PCA(n_components=3)
    comps3d = pca3d.fit_transform(feats)
    var3d = pca3d.explained_variance_ratio_

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(comps3d[:,0], comps3d[:,1], comps3d[:,2], c=timesteps, cmap='viridis', s=4, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='timestep')
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title('Visual Head PCA 3D by timestep')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'visual_pca_3d_by_timestep.png'), dpi=200)
    plt.close()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(comps3d[:,0], comps3d[:,1], comps3d[:,2], c=labels_rep, cmap='tab10', s=4, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='ball count')
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title('Visual Head PCA 3D by label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'visual_pca_3d_by_label.png'), dpi=200)
    plt.close()

    # Save PCA CSVs
    _save_pca_csv(os.path.join(save_dir, 'visual_pca2d.csv'), sample_ids, timesteps, labels_rep,
                  comps2d, ["sample_id","timestep","label","pc1","pc2"])
    _save_pca_csv(os.path.join(save_dir, 'visual_pca3d.csv'), sample_ids, timesteps, labels_rep,
                  comps3d, ["sample_id","timestep","label","pc1","pc2","pc3"])

    # 2D t-SNE (can be slower; keep params moderate)
    try:
        tsne = TSNE(n_components=2, init='pca', perplexity=min(30, max(5, (N*T)//50)), learning_rate='auto', n_iter=1000, random_state=42)
        tsne_2d = tsne.fit_transform(feats)

        plt.figure(figsize=(8,6))
        sc = plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c=timesteps, cmap='viridis', s=6)
        plt.colorbar(sc, label='timestep (0-based)')
        plt.title('Visual Head t-SNE 2D by timestep')
        plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2'); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'visual_tsne_2d_by_timestep.png'), dpi=200)
        plt.close()

        # Save t-SNE CSV
        _save_pca_csv(os.path.join(save_dir, 'visual_tsne2d.csv'), sample_ids, timesteps, labels_rep,
                  tsne_2d, ["sample_id","timestep","label","tsne1","tsne2"])

        plt.figure(figsize=(8,6))
        sc = plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c=labels_rep, cmap='tab10', s=6)
        plt.colorbar(sc, label='ball count')
        plt.title('Visual Head t-SNE 2D by label')
        plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2'); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'visual_tsne_2d_by_label.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"⚠ t-SNE 可视化失败: {e}")


def plot_lstm_pca(hidden_btH: np.ndarray, labels: np.ndarray, save_dir: str, layer_name: str = "Layer1"):
    """2D + 3D PCA scatter plots"""
    os.makedirs(save_dir, exist_ok=True)
    N, T, H = hidden_btH.shape
    feats = hidden_btH.reshape(N*T, H)
    timesteps = np.repeat(np.arange(T), N)
    labels_rep = np.repeat(labels, T)
    sample_ids = np.repeat(np.arange(N), T)

    # 2D PCA
    pca2d = PCA(n_components=2)
    comps2d = pca2d.fit_transform(feats)
    var2d = pca2d.explained_variance_ratio_

    # by timestep (2D)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(comps2d[:,0], comps2d[:,1], c=timesteps, cmap='viridis', s=6)
    plt.colorbar(sc, label='timestep (0-based)')
    plt.title(f'{layer_name} LSTM Hidden PCA 2D by timestep (PC1 {var2d[0]*100:.1f}%, PC2 {var2d[1]*100:.1f}%)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_lstm_pca_2d_by_timestep.png'), dpi=200)
    plt.close()

    # by label (2D)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(comps2d[:,0], comps2d[:,1], c=labels_rep, cmap='tab10', s=6)
    plt.colorbar(sc, label='ball count')
    plt.title(f'{layer_name} LSTM Hidden PCA 2D by label (sequence label)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_lstm_pca_2d_by_label.png'), dpi=200)
    plt.close()

    # 3D PCA
    pca3d = PCA(n_components=3)
    comps3d = pca3d.fit_transform(feats)
    var3d = pca3d.explained_variance_ratio_

    # by timestep (3D)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(comps3d[:,0], comps3d[:,1], comps3d[:,2], c=timesteps, cmap='viridis', s=4, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='timestep')
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Hidden PCA 3D by timestep')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_lstm_pca_3d_by_timestep.png'), dpi=200)
    plt.close()

    # by label (3D)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(comps3d[:,0], comps3d[:,1], comps3d[:,2], c=labels_rep, cmap='tab10', s=4, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='ball count')
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Hidden PCA 3D by label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_lstm_pca_3d_by_label.png'), dpi=200)
    plt.close()
    
    # Save CSVs
    _save_pca_csv(os.path.join(save_dir, f"{layer_name.lower()}_pca2d.csv"), sample_ids, timesteps, labels_rep,
                  comps2d, ["sample_id","timestep","label","pc1","pc2"])
    _save_pca_csv(os.path.join(save_dir, f"{layer_name.lower()}_pca3d.csv"), sample_ids, timesteps, labels_rep,
                  comps3d, ["sample_id","timestep","label","pc1","pc2","pc3"])

    return pca2d, pca3d, comps2d, comps3d, var2d, var3d


def plot_lstm_trajectories(hidden_btH: np.ndarray, labels: np.ndarray, save_dir: str, 
                           pca2d_model=None, pca3d_model=None, layer_name: str = "Layer1",
                           overlay_means: bool = True, save_means_only: bool = True,
                           overlay_variance_2d: bool = True, variance_alpha: float = 0.12):
    """2D + 3D trajectory stacking: each line = one sample's path in PCA space"""
    os.makedirs(save_dir, exist_ok=True)
    N, T, H = hidden_btH.shape
    
    if pca2d_model is None:
        feats = hidden_btH.reshape(N*T, H)
        pca2d_model = PCA(n_components=2)
        pca2d_model.fit(feats)
    if pca3d_model is None:
        feats = hidden_btH.reshape(N*T, H)
        pca3d_model = PCA(n_components=3)
        pca3d_model.fit(feats)
    
    var2d = pca2d_model.explained_variance_ratio_
    var3d = pca3d_model.explained_variance_ratio_

    # Compute per-sample trajectories
    traj2d = np.array([pca2d_model.transform(hidden_btH[i]) for i in range(N)])  # [N, T, 2]
    traj3d = np.array([pca3d_model.transform(hidden_btH[i]) for i in range(N)])  # [N, T, 3]
    labels_arr = np.asarray(labels)
    
    # === 2D Trajectory Plots ===
    # by label
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.arange(11) / 10)
    for i in range(N):
        label = int(labels[i])
        color = colors[label % 10]
        ax.plot(traj2d[i, :, 0], traj2d[i, :, 1], color=color, alpha=0.4, linewidth=1.5)
        ax.scatter(traj2d[i, 0, 0], traj2d[i, 0, 1], color=color, s=30, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.scatter(traj2d[i, -1, 0], traj2d[i, -1, 1], color=color, s=80, marker='*', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i % 10], label=f'Count={i}') for i in range(11)]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    ax.set_xlabel(f'PC1 ({var2d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var2d[1]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Trajectories 2D by Label (N={N})')
    ax.grid(alpha=0.3)

    # Overlay per-class mean trajectories (2D)
    if overlay_means:
        from matplotlib.patches import Ellipse
        for c in range(11):
            idxs = np.where(labels_arr == c)[0]
            if idxs.size == 0:
                continue
            mean_traj = traj2d[idxs].mean(axis=0)  # [T, 2]
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=colors[c % 10], linewidth=3, alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], color=colors[c % 10], s=60, marker='o', edgecolor='black', linewidth=0.6, zorder=20)
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=colors[c % 10], s=120, marker='*', edgecolor='black', linewidth=0.6, zorder=20)

            # Per-timestep covariance ellipses (1σ) as variance shading (2D only)
            if overlay_variance_2d and idxs.size >= 2:
                # Draw an ellipse at each timestep around the mean, axis=1σ from covariance
                for t in range(T):
                    pts = traj2d[idxs, t, :]  # [Nc, 2]
                    if pts.shape[0] < 2:
                        continue
                    cov = np.cov(pts.T)
                    # Numerical safety: ensure covariance is positive semidefinite
                    try:
                        eigvals, eigvecs = np.linalg.eigh(cov)
                    except np.linalg.LinAlgError:
                        continue
                    eigvals = np.clip(eigvals, 1e-9, None)
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    width = 2.0 * np.sqrt(eigvals[0])  # 1σ along first PC
                    height = 2.0 * np.sqrt(eigvals[1]) # 1σ along second PC
                    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
                    ell = Ellipse(xy=(mean_traj[t,0], mean_traj[t,1]),
                                  width=width, height=height, angle=angle,
                                  facecolor=colors[c % 10], edgecolor='none', alpha=variance_alpha, zorder=5)
                    ax.add_patch(ell)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_trajectories_2d_by_label.png'), dpi=200)
    plt.close()

    # by time
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(N):
        time_colors = plt.cm.viridis(np.linspace(0, 1, T))
        for t in range(T-1):
            ax.plot(traj2d[i, t:t+2, 0], traj2d[i, t:t+2, 1], color=time_colors[t], linewidth=2, alpha=0.7)
        ax.scatter(traj2d[i, 0, 0], traj2d[i, 0, 1], color='green', s=40, marker='o', edgecolor='black', linewidth=0.5, zorder=10)
        ax.scatter(traj2d[i, -1, 0], traj2d[i, -1, 1], color='red', s=100, marker='*', edgecolor='black', linewidth=0.5, zorder=10)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=T-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Timestep')
    ax.set_xlabel(f'PC1 ({var2d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var2d[1]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Trajectories 2D by Time (N={N})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_trajectories_2d_by_time.png'), dpi=200)
    plt.close()

    # === 3D Trajectory Plots ===
    from mpl_toolkits.mplot3d import Axes3D
    
    # by label
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        label = int(labels[i])
        color = colors[label % 10]
        ax.plot(traj3d[i, :, 0], traj3d[i, :, 1], traj3d[i, :, 2], color=color, alpha=0.4, linewidth=1.5)
        ax.scatter(traj3d[i, 0, 0], traj3d[i, 0, 1], traj3d[i, 0, 2], color=color, s=30, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.scatter(traj3d[i, -1, 0], traj3d[i, -1, 1], traj3d[i, -1, 2], color=color, s=80, marker='*', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    legend_elements = [Patch(facecolor=colors[i % 10], label=f'Count={i}') for i in range(11)]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Trajectories 3D by Label (N={N})')
    ax.view_init(elev=20, azim=45)

    # Overlay per-class mean trajectories (3D)
    if overlay_means:
        for c in range(11):
            idxs = np.where(labels_arr == c)[0]
            if idxs.size == 0:
                continue
            mean_traj3 = traj3d[idxs].mean(axis=0)  # [T, 3]
            ax.plot(mean_traj3[:, 0], mean_traj3[:, 1], mean_traj3[:, 2], color=colors[c % 10], linewidth=3, alpha=0.9)
            ax.scatter(mean_traj3[0, 0], mean_traj3[0, 1], mean_traj3[0, 2], color=colors[c % 10], s=60, marker='o', edgecolor='black', linewidth=0.6)
            ax.scatter(mean_traj3[-1, 0], mean_traj3[-1, 1], mean_traj3[-1, 2], color=colors[c % 10], s=120, marker='*', edgecolor='black', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_trajectories_3d_by_label.png'), dpi=200)
    plt.close()

    # by time
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    time_colors = plt.cm.viridis(np.linspace(0, 1, T))
    for i in range(N):
        for t in range(T-1):
            ax.plot(traj3d[i, t:t+2, 0], traj3d[i, t:t+2, 1], traj3d[i, t:t+2, 2], 
                   color=time_colors[t], linewidth=2, alpha=0.7)
        ax.scatter(traj3d[i, 0, 0], traj3d[i, 0, 1], traj3d[i, 0, 2], color='green', s=40, marker='o', edgecolor='black', linewidth=0.5, zorder=10)
        ax.scatter(traj3d[i, -1, 0], traj3d[i, -1, 1], traj3d[i, -1, 2], color='red', s=100, marker='*', edgecolor='black', linewidth=0.5, zorder=10)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=T-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Timestep', shrink=0.8)
    ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
    ax.set_title(f'{layer_name} LSTM Trajectories 3D by Time (N={N})')
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_trajectories_3d_by_time.png'), dpi=200)
    plt.close()

    print(f"✓ {layer_name}轨迹堆叠已保存 (2D+3D)")

    # Optional: save means-only figures for clarity
    if save_means_only and overlay_means:
        # 2D means-only
        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(11):
            idxs = np.where(labels_arr == c)[0]
            if idxs.size == 0:
                continue
            mean_traj = traj2d[idxs].mean(axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=colors[c % 10], linewidth=3)
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], color=colors[c % 10], s=60, marker='o', edgecolor='black', linewidth=0.6)
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=colors[c % 10], s=120, marker='*', edgecolor='black', linewidth=0.6)
        legend_elements = [Patch(facecolor=colors[i % 10], label=f'Count={i}') for i in range(11)]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        ax.set_xlabel(f'PC1 ({var2d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var2d[1]*100:.1f}%)')
        ax.set_title(f'{layer_name} Class-Mean Trajectories 2D')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_mean_trajectories_2d.png'), dpi=200)
        plt.close()

        # 3D means-only
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for c in range(11):
            idxs = np.where(labels_arr == c)[0]
            if idxs.size == 0:
                continue
            mean_traj3 = traj3d[idxs].mean(axis=0)
            ax.plot(mean_traj3[:, 0], mean_traj3[:, 1], mean_traj3[:, 2], color=colors[c % 10], linewidth=3)
            ax.scatter(mean_traj3[0, 0], mean_traj3[0, 1], mean_traj3[0, 2], color=colors[c % 10], s=60, marker='o', edgecolor='black', linewidth=0.6)
            ax.scatter(mean_traj3[-1, 0], mean_traj3[-1, 1], mean_traj3[-1, 2], color=colors[c % 10], s=120, marker='*', edgecolor='black', linewidth=0.6)
        legend_elements = [Patch(facecolor=colors[i % 10], label=f'Count={i}') for i in range(11)]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        ax.set_xlabel(f'PC1 ({var3d[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var3d[1]*100:.1f}%)'); ax.set_zlabel(f'PC3 ({var3d[2]*100:.1f}%)')
        ax.set_title(f'{layer_name} Class-Mean Trajectories 3D')
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name.lower()}_mean_trajectories_3d.png'), dpi=200)
        plt.close()



def compute_number_line_rsa(hidden_btH: np.ndarray,
                            labels: np.ndarray,
                            save_dir: str,
                            layer_name: str = "Layer1") -> dict:
    """
    Compute representational similarity analysis to detect spatial number line.
    Tests if representational distances correlate with numerical distances.
    
    Args:
        hidden_btH: [N_samples, T_timesteps, H_hidden_units] LSTM hidden states
        labels: [N_samples] ball count labels (0-10)
        save_dir: directory to save outputs
        layer_name: name of the layer (for output filenames)
    
    Returns:
        dict with:
        - 'rdm': [11, 11] representational dissimilarity matrix
        - 'correlation': Spearman correlation coefficient
        - 'p_value': p-value of correlation
        - 'mean_representations': [11, H] mean hidden states per count
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    
    os.makedirs(save_dir, exist_ok=True)
    
    N, T, H = hidden_btH.shape
    print(f"\n{'='*60}")
    print(f"计算{layer_name}的数字线RSA")
    print(f"样本数={N}, 序列长={T}, 隐藏单元数={H}")
    print(f"{'='*60}")
    
    # Step 1: Extract final timestep and compute mean representations per count
    final_hidden = hidden_btH[:, -1, :]  # [N, H]
    mean_reps = np.zeros((11, H))
    counts_per_class = np.zeros(11)
    
    for count in range(11):
        mask = (labels == count)
        if mask.sum() > 0:
            mean_reps[count] = final_hidden[mask].mean(axis=0)
            counts_per_class[count] = mask.sum()
    
    print(f"样本分布: {counts_per_class.astype(int)}")
    
    # Step 2: Compute representational dissimilarity matrix (RDM)
    # Using Euclidean distance between mean representations
    distances = pdist(mean_reps, metric='euclidean')
    rdm = squareform(distances)
    
    # Normalize by max distance
    rdm_norm = rdm / (rdm.max() + 1e-8)
    
    print(f"RDM距离范围: [{rdm.min():.4f}, {rdm.max():.4f}]")
    
    # Step 3: Compute numerical distance matrix
    numerical_dist = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            numerical_dist[i, j] = abs(i - j)
    
    # Step 4: Statistical test - Spearman correlation
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(11, k=1)
    rdm_flat = rdm_norm[triu_idx]
    num_dist_flat = numerical_dist[triu_idx]
    
    # Compute Spearman correlation
    corr, p_val = spearmanr(num_dist_flat, rdm_flat)
    
    print(f"Spearman相关系数: r={corr:.4f}, p={p_val:.6f}")
    if p_val < 0.05:
        print(f"✓ 显著相关 (p < 0.05)")
    else:
        print(f"✗ 不显著相关 (p >= 0.05)")
    
    # ===== Visualization 1: RDM Heatmap =====
    print(f"\n生成RDM热力图...")
    fig, ax = plt.subplots(figsize=(10, 9))
    
    im = ax.imshow(rdm_norm, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i in range(11):
        for j in range(11):
            text = ax.text(j, i, f'{rdm_norm[i, j]:.2f}',
                          ha="center", va="center", color="w" if rdm_norm[i, j] > 0.5 else "black",
                          fontsize=8)
    
    ax.set_xticks(np.arange(11))
    ax.set_yticks(np.arange(11))
    ax.set_xticklabels(np.arange(11))
    ax.set_yticklabels(np.arange(11))
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{layer_name} Representational Dissimilarity Matrix', 
                fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Normalized Distance')
    
    plt.tight_layout()
    rdm_path = os.path.join(save_dir, f'{layer_name.lower()}_rdm.png')
    plt.savefig(rdm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ RDM热力图已保存: {rdm_path}")
    
    # ===== Visualization 2: Correlation Scatter Plot =====
    print(f"生成数字线相关散点图...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    ax.scatter(num_dist_flat, rdm_flat, s=100, alpha=0.6, edgecolor='black', linewidth=1.5)
    
    # Add regression line
    z = np.polyfit(num_dist_flat, rdm_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(num_dist_flat.min(), num_dist_flat.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2.5, alpha=0.8, label=f'Linear fit')
    
    ax.set_xlabel('Numerical Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Representational Distance', fontsize=12, fontweight='bold')
    ax.set_title(f'{layer_name} Number Line Effect\n(ρ={corr:.3f}, p={p_val:.4f})', 
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    corr_path = os.path.join(save_dir, f'{layer_name.lower()}_number_line_correlation.png')
    plt.savefig(corr_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 相关散点图已保存: {corr_path}")
    
    # ===== Data Export =====
    print(f"导出数据到CSV...")
    
    # Export RDM as CSV with count labels
    rdm_csv = os.path.join(save_dir, f'{layer_name.lower()}_rdm.csv')
    counts = np.arange(11)
    # Add count labels as first row and first column
    rdm_with_labels = np.column_stack([counts, rdm_norm])
    header = 'count,' + ','.join([f'count_{c}' for c in counts])
    np.savetxt(rdm_csv, rdm_with_labels, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f"✓ RDM已保存: {rdm_csv}")
    
    # Export correlation results to text file
    results_txt = os.path.join(save_dir, f'{layer_name.lower()}_rsa_results.txt')
    with open(results_txt, 'w') as f:
        f.write(f"Representational Similarity Analysis (RSA) Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Sample distribution: {', '.join([f'{int(c)}' for c in counts_per_class])}\n\n")
        f.write(f"Number Line Effect:\n")
        f.write(f"  Spearman correlation (ρ): {corr:.6f}\n")
        f.write(f"  p-value: {p_val:.6f}\n")
        f.write(f"  Significant: {'Yes (p < 0.05)' if p_val < 0.05 else 'No (p >= 0.05)'}\n\n")
        f.write(f"Interpretation:\n")
        if p_val < 0.05:
            if corr > 0:
                f.write(f"  ✓ 存在显著的正相关：数值距离越大，表征差异越大\n")
                f.write(f"    这表明{layer_name}的单元编码了一个数字线\n")
            else:
                f.write(f"  ? 存在显著的负相关（意外）\n")
        else:
            f.write(f"  ✗ 没有显著相关：表征距离与数值距离无关\n")
            f.write(f"    {layer_name}不编码空间排序的数字\n")
    
    print(f"✓ 结果已保存: {results_txt}")
    
    # Export mean representations
    mean_reps_csv = os.path.join(save_dir, f'{layer_name.lower()}_mean_representations.csv')
    mean_reps_with_labels = np.column_stack([np.arange(11), mean_reps])
    header_mean = 'count,' + ','.join([f'unit_{h}' for h in range(H)])
    np.savetxt(mean_reps_csv, mean_reps_with_labels, delimiter=',', 
              header=header_mean, comments='', fmt='%.6f')
    print(f"✓ 平均表征已保存: {mean_reps_csv}")
    
    result = {
        'rdm': rdm_norm,
        'correlation': float(corr),
        'p_value': float(p_val),
        'mean_representations': mean_reps,
    }
    
    return result


def analyze_rotational_dynamics(hidden_btH: np.ndarray,
                                labels: np.ndarray,
                                pca2d_model,
                                save_dir: str,
                                layer_name: str = "Layer1") -> dict:
    """
    Analyze whether LSTM trajectories exhibit rotational dynamics.
    
    Rotational dynamics occur when velocity vectors are perpendicular to position vectors,
    indicating the state space evolution follows circular/elliptical paths rather than 
    straight lines. This is often observed in neural systems performing rhythmic or 
    sequential computations.
    
    Args:
        hidden_btH: [N_samples, T_timesteps, H_hidden_units] LSTM hidden states
        labels: [N_samples] ball count labels (0-10)
        pca2d_model: Fitted PCA model with 2 components
        save_dir: directory to save outputs
        layer_name: name of the layer (for output filenames)
    
    Returns:
        dict with:
        - 'mean_trajectories_2d': [11, T, 2] mean trajectories in 2D PCA space
        - 'velocities_2d': [11, T-1, 2] velocity vectors
        - 'angles': array of all angles between position and velocity
        - 'rotation_score': float (0-1), 1=strong rotation, 0=no rotation
        - 'mean_angle': float, mean angle in degrees
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, T, H = hidden_btH.shape
    print(f"\n{'='*60}")
    print(f"旋转动力学分析 - {layer_name}")
    print(f"{'='*60}")
    print(f"数据形状: N={N}, T={T}, H={H}")
    
    # ===== Step 1: Project trajectories to 2D PCA space =====
    print(f"步骤1: 投影轨迹到2D PCA空间...")
    traj_2d = np.zeros((N, T, 2))
    for i in range(N):
        traj_2d[i] = pca2d_model.transform(hidden_btH[i])  # [T, 2]
    print(f"✓ 轨迹形状: {traj_2d.shape}")
    
    # ===== Step 2: Compute per-class mean trajectories =====
    print(f"步骤2: 计算每个类别的平均轨迹...")
    mean_traj_2d = np.zeros((11, T, 2))
    counts_per_class = np.zeros(11)
    for c in range(11):
        mask = (labels == c)
        if mask.sum() > 0:
            mean_traj_2d[c] = traj_2d[mask].mean(axis=0)  # [T, 2]
            counts_per_class[c] = mask.sum()
    print(f"✓ 平均轨迹形状: {mean_traj_2d.shape}")
    print(f"  每类样本数: {counts_per_class.astype(int)}")
    
    # ===== Step 3: Compute velocity (derivative) =====
    print(f"步骤3: 计算速度向量...")
    velocity_2d = np.zeros((11, T-1, 2))
    for c in range(11):
        velocity_2d[c] = np.diff(mean_traj_2d[c], axis=0)  # [T-1, 2]
    print(f"✓ 速度形状: {velocity_2d.shape}")
    
    # ===== Step 4-5: Check perpendicularity (rotation indicator) =====
    print(f"步骤4-5: 检查垂直性（旋转指标）...")
    angles = []
    
    for c in range(11):
        if counts_per_class[c] == 0:
            continue
        for t in range(T-1):
            position = mean_traj_2d[c, t, :]  # [2]
            velocity = velocity_2d[c, t, :]    # [2]
            
            pos_mag = np.linalg.norm(position)
            vel_mag = np.linalg.norm(velocity)
            
            # Avoid division by zero
            if pos_mag < 1e-8 or vel_mag < 1e-8:
                continue
            
            # Compute angle between position and velocity
            dot_product = np.dot(position, velocity)
            cos_theta = dot_product / (pos_mag * vel_mag)
            # Clamp to [-1, 1] to avoid numerical errors in arccos
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta_rad = np.arccos(cos_theta)
            theta_deg = np.degrees(theta_rad)
            angles.append(theta_deg)
    
    angles = np.array(angles)
    print(f"✓ 计算了 {len(angles)} 个角度")
    
    # ===== Step 8: Quantitative metric =====
    print(f"步骤8: 计算旋转得分...")
    deviation_from_90 = np.abs(angles - 90.0)
    mean_deviation = deviation_from_90.mean()
    rotation_score = 1.0 - (mean_deviation / 90.0)
    rotation_score = np.clip(rotation_score, 0.0, 1.0)
    
    mean_angle = angles.mean()
    std_angle = angles.std()
    
    print(f"✓ 平均角度: {mean_angle:.2f}° (std={std_angle:.2f}°)")
    print(f"✓ 平均偏离90°: {mean_deviation:.2f}°")
    print(f"✓ 旋转得分: {rotation_score:.4f} (1=强旋转, 0=无旋转)")
    
    # ===== Step 6: Visualization - Panel A (Trajectories with Velocity Vectors) =====
    print(f"步骤6: 绘制轨迹与速度向量...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = plt.cm.tab10
    colors = [cmap(c / 10) for c in range(11)]
    
    for c in range(11):
        if counts_per_class[c] == 0:
            continue
        
        traj = mean_traj_2d[c]  # [T, 2]
        vel = velocity_2d[c]     # [T-1, 2]
        
        # Plot trajectory line
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[c], 
               linewidth=2, alpha=0.7, label=f'Count {c}')
        
        # Add arrows every 2 timesteps
        for t in range(0, T-1, 2):
            pos = traj[t]
            v = vel[t]
            # Scale arrow length for visibility
            arrow_scale = 0.3
            ax.arrow(pos[0], pos[1], v[0]*arrow_scale, v[1]*arrow_scale,
                    head_width=0.15, head_length=0.1, fc=colors[c], ec=colors[c],
                    alpha=0.6, linewidth=1.5)
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', 
                  edgecolors='black', linewidths=1.5, zorder=10, alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', 
                  edgecolors='black', linewidths=1.5, zorder=10, alpha=0.8)
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_title(f'{layer_name} Mean Trajectories with Velocity Vectors\n(Green=Start, Red=End)', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.3)
    
    plt.tight_layout()
    traj_path = os.path.join(save_dir, f'{layer_name.lower()}_rotation_analysis_velocities.png')
    plt.savefig(traj_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 轨迹与速度向量图已保存: {traj_path}")
    
    # ===== Step 7: Visualization - Panel B (Angle Distribution) =====
    print(f"步骤7: 绘制角度分布...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(angles, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(90, color='red', linestyle='--', linewidth=2.5, label='90° (Perfect Rotation)')
    ax.axvline(mean_angle, color='orange', linestyle='--', linewidth=2, label=f'Mean Angle ({mean_angle:.1f}°)')
    
    ax.set_xlabel('Angle between Position and Velocity (degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'{layer_name} Perpendicularity Check\n(Rotation if θ≈90°, Score={rotation_score:.3f})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'Mean: {mean_angle:.2f}°\nStd: {std_angle:.2f}°\nDeviation from 90°: {mean_deviation:.2f}°'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    angle_path = os.path.join(save_dir, f'{layer_name.lower()}_rotation_angle_distribution.png')
    plt.savefig(angle_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 角度分布图已保存: {angle_path}")
    
    # ===== Step 9: Export data =====
    print(f"步骤9: 导出数据...")
    
    # Export mean trajectories
    traj_csv = os.path.join(save_dir, f'{layer_name.lower()}_mean_trajectories_2d.csv')
    # Reshape [11, T, 2] -> rows with format: count, timestep, pc1, pc2
    traj_rows = []
    for c in range(11):
        for t in range(T):
            traj_rows.append([c, t, mean_traj_2d[c, t, 0], mean_traj_2d[c, t, 1]])
    traj_data = np.array(traj_rows)
    np.savetxt(traj_csv, traj_data, delimiter=',', 
              header='count,timestep,pc1,pc2', comments='', fmt='%.6f')
    print(f"✓ 平均轨迹已保存: {traj_csv}")
    
    # Export velocities
    vel_csv = os.path.join(save_dir, f'{layer_name.lower()}_velocities_2d.csv')
    vel_rows = []
    for c in range(11):
        for t in range(T-1):
            vel_rows.append([c, t, velocity_2d[c, t, 0], velocity_2d[c, t, 1]])
    vel_data = np.array(vel_rows)
    np.savetxt(vel_csv, vel_data, delimiter=',', 
              header='count,timestep,velocity_pc1,velocity_pc2', comments='', fmt='%.6f')
    print(f"✓ 速度向量已保存: {vel_csv}")
    
    # Export angles and rotation score
    results_txt = os.path.join(save_dir, f'{layer_name.lower()}_rotation_results.txt')
    with open(results_txt, 'w') as f:
        f.write(f"Rotational Dynamics Analysis Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Sample distribution: {', '.join([f'{int(c)}' for c in counts_per_class])}\n\n")
        f.write(f"Angle Statistics:\n")
        f.write(f"  Mean angle: {mean_angle:.4f}°\n")
        f.write(f"  Std angle: {std_angle:.4f}°\n")
        f.write(f"  Mean deviation from 90°: {mean_deviation:.4f}°\n")
        f.write(f"  Total angles computed: {len(angles)}\n\n")
        f.write(f"Rotation Score: {rotation_score:.6f}\n")
        f.write(f"  (1.0 = perfect rotation, 0.0 = no rotation)\n\n")
        f.write(f"Interpretation:\n")
        if rotation_score > 0.7:
            f.write(f"  ✓ 强旋转动力学：角度接近90°\n")
            f.write(f"    {layer_name}的轨迹展示出旋转结构\n")
        elif rotation_score > 0.4:
            f.write(f"  ~ 中等旋转动力学：部分旋转特征\n")
            f.write(f"    {layer_name}的轨迹显示一些旋转倾向\n")
        else:
            f.write(f"  ✗ 无明显旋转：轨迹主要是直线运动\n")
            f.write(f"    {layer_name}不展示旋转动力学\n")
    print(f"✓ 结果已保存: {results_txt}")
    
    # Export angles to CSV
    angles_csv = os.path.join(save_dir, f'{layer_name.lower()}_angles.csv')
    np.savetxt(angles_csv, angles, delimiter=',', 
              header='angle_degrees', comments='', fmt='%.6f')
    print(f"✓ 角度数据已保存: {angles_csv}")
    
    result = {
        'mean_trajectories_2d': mean_traj_2d,
        'velocities_2d': velocity_2d,
        'angles': angles,
        'rotation_score': float(rotation_score),
        'mean_angle': float(mean_angle),
    }
    
    return result


def perform_jPCA_analysis(hidden_btH: np.ndarray,
                         labels: np.ndarray,
                         save_dir: str,
                         layer_name: str = "Layer1",
                         pca_n_components: int = 6) -> dict:
    """
    Perform jPCA (joint Principal Component Analysis) to identify rotational dynamics.
    
    Based on Churchland et al. (2012) Nature.
    
    jPCA differs from standard PCA by:
    1. Fitting a linear dynamical system: dX/dt = M × X
    2. Extracting the skew-symmetric (rotational) component of M
    3. Finding optimal rotational axes (jPCs) via eigendecomposition
    4. Projecting trajectories onto the jPC plane (not variance-maximizing PCA axes!)
    
    Args:
        hidden_btH: [N, T, H] LSTM hidden states
        labels: [N] ball count labels (0-10)
        save_dir: output directory
        layer_name: layer name for labeling
        pca_n_components: number of PCA dimensions for preprocessing (default 6)
    
    Returns:
        dict with:
        - 'jPCs': [pca_n_components, 2] rotational principal components
        - 'M_skew': [pca_n_components, pca_n_components] rotation matrix
        - 'eigenvalues': complex eigenvalues
        - 'rotation_frequency_omega': float (rad/timestep)
        - 'rotation_variance_fraction': float (0-1)
        - 'dynamics_fit_R2': float (quality of fit)
        - 'rotation_quality': float (perpendicularity score 0-1)
        - 'trajectories_jPCA': dict of [T, 2] trajectories in jPC space
    """
    import json
    from sklearn.decomposition import PCA
    
    os.makedirs(save_dir, exist_ok=True)
    
    N, T, H = hidden_btH.shape
    print(f"\n{'='*60}")
    print(f"jPCA分析 - {layer_name}")
    print(f"{'='*60}")
    print(f"数据形状: N={N}, T={T}, H={H}")
    print(f"目标PCA维度: K={pca_n_components}")
    
    # ===== Step 1: PCA Preprocessing (Dimensionality Reduction) =====
    print(f"\n步骤1: PCA预处理降维 (H={H} → K={pca_n_components})...")
    X_all = hidden_btH.reshape(N*T, H)
    
    pca_prep = PCA(n_components=pca_n_components)
    pca_prep.fit(X_all)
    
    variance_explained = pca_prep.explained_variance_ratio_.sum()
    print(f"✓ PCA解释方差: {variance_explained:.1%}")
    
    # Project all trajectories to K-dimensional PCA space
    X_pca = np.array([pca_prep.transform(hidden_btH[i]) for i in range(N)])  # [N, T, K]
    print(f"✓ 投影后形状: {X_pca.shape}")
    
    # ===== Step 2: Compute Per-Class Mean Trajectories in PCA Space =====
    print(f"\n步骤2: 计算每类平均轨迹...")
    mean_traj = {}
    counts_per_class = []
    for c in range(11):
        mask = (labels == c)
        if mask.sum() > 0:
            mean_traj[c] = X_pca[mask].mean(axis=0)  # [T, K]
            counts_per_class.append((c, mask.sum()))
    print(f"✓ 有效类别数: {len(mean_traj)}")
    print(f"  样本分布: {', '.join([f'{c}:{n}' for c, n in counts_per_class])}")
    
    # ===== Step 3: Compute Velocities (Temporal Derivatives) =====
    print(f"\n步骤3: 计算速度向量 (时间导数)...")
    velocity = {}
    for c in mean_traj.keys():
        velocity[c] = np.diff(mean_traj[c], axis=0)  # [T-1, K]
    print(f"✓ 速度形状: [T-1={T-1}, K={pca_n_components}]")
    
    # ===== Step 4: Prepare Data for Dynamics Fitting =====
    print(f"\n步骤4: 准备动力学拟合数据...")
    X_fit = np.vstack([mean_traj[c][:-1] for c in mean_traj.keys()])  # [M, K]
    dX_fit = np.vstack([velocity[c] for c in velocity.keys()])        # [M, K]
    M_samples = X_fit.shape[0]
    print(f"✓ 拟合数据点: M={M_samples} (条件数×时间步)")
    
    # ===== Step 5: Fit Linear Dynamical System =====
    print(f"\n步骤5: 拟合线性动力系统 dX/dt = M @ X...")
    # Solve: M = dX_fit.T @ pinv(X_fit.T)
    M = dX_fit.T @ np.linalg.pinv(X_fit.T)  # [K, K]
    
    # Evaluate fit quality (R-squared)
    dX_pred = (M @ X_fit.T).T
    residual = dX_fit - dX_pred
    total_variance = np.var(dX_fit)
    residual_variance = np.var(residual)
    R2 = 1 - residual_variance / total_variance
    
    print(f"✓ 动力系统矩阵M: {M.shape}")
    print(f"✓ 拟合质量 R² = {R2:.4f}")
    
    # ===== Step 6: Extract Skew-Symmetric (Rotation) Component =====
    print(f"\n步骤6: 提取偏对称旋转分量...")
    # Decompose M into symmetric and skew-symmetric parts
    M_symmetric = (M + M.T) / 2   # expansion/contraction
    M_skew = (M - M.T) / 2        # ROTATION (this is what we want!)
    
    # Verify skew-symmetry: M_skew should equal -M_skew.T
    skew_error = np.max(np.abs(M_skew + M_skew.T))
    print(f"✓ 偏对称验证: max|M_skew + M_skew.T| = {skew_error:.2e}")
    assert skew_error < 1e-10, f"M_skew必须是偏对称的，但误差为{skew_error}"
    
    # Compute how much variance rotation explains
    rot_dynamics = (M_skew @ X_fit.T).T
    rot_var = float(np.var(rot_dynamics))
    total_var = float(np.var(dX_fit))
    if not np.isfinite(total_var) or total_var <= 1e-12 or not np.isfinite(rot_var):
        rotation_variance_fraction = 0.0
    else:
        rotation_variance_fraction = rot_var / total_var
    rotation_variance_fraction = float(np.clip(rotation_variance_fraction, 0.0, 1.0))
    
    print(f"✓ 旋转解释的动力学方差: {rotation_variance_fraction:.1%}")
    
    # ===== Step 7: Eigendecomposition to Find jPCs =====
    print(f"\n步骤7: 特征分解寻找旋转轴 (jPCs)...")
    # For skew-symmetric matrices, eigenvalues are purely imaginary (conjugate pairs)
    eigenvalues, eigenvectors = np.linalg.eig(M_skew)
    
    # Sort by magnitude of imaginary part (rotation frequency)
    idx_sorted = np.argsort(np.abs(eigenvalues.imag))[::-1]
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors = eigenvectors[:, idx_sorted]
    
    print(f"✓ 特征值（虚部，旋转频率）:")
    for i in range(min(3, len(eigenvalues))):
        print(f"    λ{i}: {eigenvalues[i].real:.4f} + {eigenvalues[i].imag:.4f}i")
    
    # Extract top 2 eigenvectors (largest rotation)
    # Use real parts (imaginary parts are conjugates)
    jPC1 = np.real(eigenvectors[:, 0])  # [K]
    jPC2 = np.imag(eigenvectors[:, 0])  # [K]
    jPCs = np.column_stack([jPC1, jPC2])  # [K, 2]
    
    # Primary rotation frequency
    omega = np.abs(eigenvalues[0].imag)
    print(f"✓ 主旋转频率 ω = {omega:.4f} rad/timestep")
    print(f"✓ jPCs形状: {jPCs.shape}")
    
    # ===== Step 8: Project Trajectories onto jPC Space =====
    print(f"\n步骤8: 将轨迹投影到jPC空间...")
    traj_jPCA = {}
    vel_jPCA = {}
    
    for c in mean_traj.keys():
        traj_jPCA[c] = mean_traj[c] @ jPCs      # [T, 2]
        vel_jPCA[c] = velocity[c] @ jPCs        # [T-1, 2]
    
    print(f"✓ jPC空间轨迹形状: [T={T}, 2]")
    
    # ===== Step 9: Quantify Rotation Quality (Perpendicularity) =====
    print(f"\n步骤9: 量化旋转质量（垂直性检查）...")
    angles = []
    
    for c in traj_jPCA.keys():
        for t in range(len(vel_jPCA[c])):
            pos = traj_jPCA[c][t]
            vel = vel_jPCA[c][t]
            
            pos_norm = np.linalg.norm(pos)
            vel_norm = np.linalg.norm(vel)
            
            if pos_norm > 1e-10 and vel_norm > 1e-10:
                cos_angle = np.dot(pos, vel) / (pos_norm * vel_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                angles.append(angle_deg)
    
    angles = np.array(angles)
    if angles.size == 0:
        mean_angle = float('nan')
        rotation_quality = 0.0
    else:
        mean_angle = float(np.mean(angles))
        rotation_quality = float(1 - np.abs(90 - mean_angle) / 90)
        rotation_quality = float(np.clip(rotation_quality, 0.0, 1.0))
    
    print(f"✓ 平均垂直角度: {mean_angle:.2f}° (理想值=90°)")
    print(f"✓ 旋转质量: {rotation_quality:.4f} (0-1)")
    
    # ===== Step 10: Visualization - jPCA Trajectories =====
    print(f"\n步骤10: 绘制jPCA轨迹...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel A: Trajectories in jPC space
    colors = plt.cm.tab10(np.arange(11) / 10)
    
    for c in sorted(traj_jPCA.keys()):
        color = colors[c % 10]
        traj = traj_jPCA[c]
        vel = vel_jPCA[c]
        
        # Plot trajectory
        axes[0].plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.5, label=f'Count={c}', alpha=0.8)
        
        # Mark start (circle) and end (star)
        axes[0].scatter(traj[0, 0], traj[0, 1], color=color, s=60, marker='o', 
                       edgecolor='black', linewidth=0.6, zorder=10)
        axes[0].scatter(traj[-1, 0], traj[-1, 1], color=color, s=120, marker='*',
                       edgecolor='black', linewidth=0.6, zorder=10)
        
        # Velocity arrows every 2 timesteps
        for t in range(0, len(vel), 2):
            if t < len(vel):
                axes[0].quiver(traj[t, 0], traj[t, 1], vel[t, 0], vel[t, 1],
                              color=color, alpha=0.5, width=0.003, scale=20)
    
    axes[0].set_xlabel('jPC1', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('jPC2', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{layer_name} jPCA Rotational Dynamics\n(ω={omega:.3f} rad/step, R²={R2:.3f})', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9, ncol=2, loc='best')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.3)
    axes[0].axvline(0, color='black', linewidth=0.5, alpha=0.3)
    axes[0].axis('equal')  # CRITICAL for visualizing rotation!
    
    # Panel B: Phase portrait (position vs velocity in jPC1)
    for c in sorted(traj_jPCA.keys()):
        color = colors[c % 10]
        axes[1].scatter(traj_jPCA[c][:-1, 0], vel_jPCA[c][:, 0],
                       color=color, s=30, alpha=0.6, label=f'Count={c}')
    
    axes[1].set_xlabel('jPC1 Position', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('jPC1 Velocity', fontsize=12, fontweight='bold')
    axes[1].set_title('Phase Portrait\n(Circular pattern indicates rotation)', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9, ncol=2, loc='best')
    
    plt.tight_layout()
    traj_plot_path = os.path.join(save_dir, f'{layer_name.lower()}_jPCA_analysis.png')
    plt.savefig(traj_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ jPCA轨迹图已保存: {traj_plot_path}")
    
    # ===== Step 11: Summary Figure (Eigenvalues & Statistics) =====
    print(f"\n步骤11: 绘制总结图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Eigenvalue spectrum (rotation frequencies)
    axes[0].bar(range(pca_n_components), np.abs(eigenvalues.imag), 
               color=['red' if i < 2 else 'steelblue' for i in range(pca_n_components)],
               edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Eigenvalue Index', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Rotation Frequency (rad/timestep)', fontsize=11, fontweight='bold')
    axes[0].set_title('Rotational Eigenvalue Spectrum\n(Red = used for jPCs)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticks(range(pca_n_components))
    
    # Panel B: Variance pie chart
    sizes = [max(0.0, rotation_variance_fraction * 100.0), max(0.0, (1.0 - rotation_variance_fraction) * 100.0)]
    colors_pie = ['#ff9999', '#66b3ff']
    if np.isfinite(sum(sizes)) and sum(sizes) > 0 and all(np.isfinite(s) for s in sizes):
        axes[1].pie(sizes, labels=['Rotational', 'Non-rotational'], autopct='%1.1f%%',
                   colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title(f'Dynamics Variance Breakdown\nRotation: {rotation_variance_fraction:.1%}', 
                         fontsize=12, fontweight='bold')
    else:
        axes[1].bar(['Rotational','Non-rotational'], sizes, color=colors_pie, edgecolor='black')
        axes[1].set_ylim(0, 100)
        axes[1].set_title('Dynamics Variance Breakdown (fallback)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Percent (%)', fontsize=11, fontweight='bold')
    
    # Panel C: Perpendicularity histogram
    axes[2].hist(angles, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[2].axvline(90, color='red', linestyle='--', linewidth=2.5, label='90° (Perfect)')
    if np.isfinite(mean_angle):
        axes[2].axvline(mean_angle, color='orange', linestyle='--', linewidth=2, label=f'Mean ({mean_angle:.1f}°)')
    axes[2].set_xlabel('Angle (degrees)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[2].set_title(f'Perpendicularity Distribution\nQuality: {rotation_quality:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_plot_path = os.path.join(save_dir, f'{layer_name.lower()}_jPCA_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ jPCA总结图已保存: {summary_plot_path}")
    
    # ===== Step 12: Export All Data =====
    print(f"\n步骤12: 导出数据...")
    
    # Save jPCs
    jpcs_path = os.path.join(save_dir, f'{layer_name.lower()}_jPCs.npy')
    np.save(jpcs_path, jPCs)
    print(f"✓ jPCs已保存: {jpcs_path}")
    
    # Save rotation matrix
    m_skew_path = os.path.join(save_dir, f'{layer_name.lower()}_M_skew.npy')
    np.save(m_skew_path, M_skew)
    print(f"✓ 旋转矩阵M_skew已保存: {m_skew_path}")
    
    # Save eigenvalues
    eigenval_path = os.path.join(save_dir, f'{layer_name.lower()}_eigenvalues.npy')
    np.save(eigenval_path, eigenvalues)
    print(f"✓ 特征值已保存: {eigenval_path}")
    
    # Save trajectories in jPC space
    traj_csv = os.path.join(save_dir, f'{layer_name.lower()}_jPCA_trajectories.csv')
    rows = []
    for c in sorted(traj_jPCA.keys()):
        for t in range(T):
            rows.append([c, t, traj_jPCA[c][t, 0], traj_jPCA[c][t, 1]])
    traj_data = np.array(rows)
    np.savetxt(traj_csv, traj_data, delimiter=',', 
              header='count,timestep,jPC1,jPC2', comments='', fmt='%.6f')
    print(f"✓ jPC轨迹已保存: {traj_csv}")
    
    # Save summary statistics as JSON
    summary = {
        'layer_name': layer_name,
        'pca_n_components': pca_n_components,
        'pca_variance_explained': float(variance_explained),
        'rotation_frequency_omega': float(omega),
        'rotation_variance_fraction': float(rotation_variance_fraction),
        'dynamics_fit_R2': float(R2),
        'rotation_quality': float(rotation_quality),
        'mean_perpendicularity_angle': float(mean_angle),
        'num_conditions': len(traj_jPCA),
        'num_timesteps': T,
    }
    json_path = os.path.join(save_dir, f'{layer_name.lower()}_jPCA_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 总结统计已保存: {json_path}")
    
    # Save detailed text report
    report_path = os.path.join(save_dir, f'{layer_name.lower()}_jPCA_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"jPCA Analysis Report - {layer_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Based on Churchland et al. (2012) Nature\n\n")
        f.write(f"Data Info:\n")
        f.write(f"  Samples: {N}\n")
        f.write(f"  Timesteps: {T}\n")
        f.write(f"  Original dimensions: {H}\n")
        f.write(f"  PCA preprocessing dimensions: {pca_n_components}\n")
        f.write(f"  PCA variance explained: {variance_explained:.2%}\n")
        f.write(f"  Number of conditions: {len(traj_jPCA)}\n\n")
        f.write(f"Dynamics Fitting:\n")
        f.write(f"  Model: dX/dt = M @ X\n")
        f.write(f"  Fit quality R²: {R2:.4f}\n")
        f.write(f"  Matrix M shape: {M.shape}\n\n")
        f.write(f"Rotational Component:\n")
        f.write(f"  Rotation explains: {rotation_variance_fraction:.2%} of dynamics variance\n")
        f.write(f"  Primary rotation frequency ω: {omega:.4f} rad/timestep\n")
        f.write(f"  Period (if cyclic): {2*np.pi/omega if omega > 1e-6 else float('inf'):.2f} timesteps\n\n")
        f.write(f"jPCs (Rotational Axes):\n")
        f.write(f"  Shape: {jPCs.shape}\n")
        f.write(f"  Top eigenvalues (imaginary parts):\n")
        for i in range(min(pca_n_components, len(eigenvalues))):
            f.write(f"    λ{i}: {eigenvalues[i].imag:.4f}i\n")
        f.write(f"\n")
        f.write(f"Rotation Quality:\n")
        f.write(f"  Mean perpendicularity angle: {mean_angle:.2f}° (ideal=90°)\n")
        f.write(f"  Rotation quality score: {rotation_quality:.4f} (0-1 scale)\n")
        f.write(f"  Interpretation: ")
        if rotation_quality > 0.7:
            f.write(f"Strong rotational dynamics detected\n")
        elif rotation_quality > 0.4:
            f.write(f"Moderate rotational dynamics\n")
        else:
            f.write(f"Weak or no rotational dynamics\n")
    print(f"✓ 详细报告已保存: {report_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ jPCA分析完成")
    print(f"  - ω = {omega:.4f} rad/step")
    print(f"  - Rotation explains {rotation_variance_fraction:.1%} variance")
    print(f"  - Fit R² = {R2:.4f}")
    print(f"  - Quality = {rotation_quality:.4f}")
    print(f"{'='*60}")
    
    # ===== Step 13: Return Results =====
    return {
        'jPCs': jPCs,
        'M_skew': M_skew,
        'eigenvalues': eigenvalues,
        'rotation_frequency_omega': float(omega),
        'rotation_variance_fraction': float(rotation_variance_fraction),
        'dynamics_fit_R2': float(R2),
        'rotation_quality': float(rotation_quality),
        'trajectories_jPCA': traj_jPCA,
    }


def analyze_number_selectivity(hidden_btH: np.ndarray, 
                               labels: np.ndarray, 
                               save_dir: str,
                               layer_name: str = "Layer1",
                               top_k: int = 50) -> dict:
    """
    Analyze number-selective units in LSTM hidden states.
    
    Args:
        hidden_btH: [N_samples, T_timesteps, H_hidden_units] LSTM hidden states
        labels: [N_samples] ball count labels (0-10)
        save_dir: directory to save outputs
        layer_name: name of the layer (for output filenames)
        top_k: number of top selective units to visualize
    
    Returns:
        dict with:
        - 'tuning_curves': [H, 11] array of mean activations per count
        - 'selectivity': [H] array of selectivity indices
        - 'top_units': list of top K unit indices
        - 'percent_selective': percentage of units above 90th percentile
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, T, H = hidden_btH.shape
    print(f"\n{'='*60}")
    print(f"分析{layer_name}的数字选择性单元")
    print(f"样本数={N}, 序列长={T}, 隐藏单元数={H}")
    print(f"{'='*60}")
    
    # Step 1: Extract final timestep activations
    final_hidden = hidden_btH[:, -1, :]  # [N, H]
    
    # Step 2: Compute tuning curves - mean activation for each unit at each count
    tuning_curves = np.zeros((H, 11))  # 11 classes (0-10)
    counts_per_class = np.zeros(11)
    
    for count in range(11):
        mask = (labels == count)
        if mask.sum() > 0:
            tuning_curves[:, count] = final_hidden[mask].mean(axis=0)
            counts_per_class[count] = mask.sum()
    
    print(f"样本分布: {counts_per_class.astype(int)}")
    
    # Step 3: Compute selectivity index - std/mean (higher = more selective)
    selectivity = np.zeros(H)
    for h in range(H):
        tc = tuning_curves[h]
        mean_act = np.abs(tc).mean() + 1e-6  # Add epsilon to avoid division by zero
        std_act = tc.std()
        selectivity[h] = std_act / mean_act
    
    # Step 4: Identify top selective units
    top_indices = np.argsort(selectivity)[::-1][:top_k]
    top_selectivity = selectivity[top_indices]
    
    # Step 5: Compute percentile threshold
    percentile_90 = np.percentile(selectivity, 90)
    percent_selective = 100.0 * (selectivity > percentile_90).sum() / H
    
    print(f"选择性得分范围: [{selectivity.min():.4f}, {selectivity.max():.4f}]")
    print(f"90%分位数阈值: {percentile_90:.4f}")
    print(f"高度选择性的单元比例: {percent_selective:.1f}%")
    print(f"前{top_k}个选择性单元的得分: {top_selectivity[:5]} ... {top_selectivity[-5:]}")
    
    # ===== Visualization 1: Tuning curves for top units =====
    print(f"\n生成前{top_k}个单元的调谐曲线...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(10, 5, hspace=0.4, wspace=0.4)
    
    for plot_idx, unit_idx in enumerate(top_indices):
        row = plot_idx // 5
        col = plot_idx % 5
        ax = fig.add_subplot(gs[row, col])
        
        tc = tuning_curves[unit_idx]
        counts = np.arange(11)
        pref_count = np.argmax(np.abs(tc))  # Preferred count (argmax of absolute activity)
        
        # Plot tuning curve
        colors = ['red' if c == pref_count else 'steelblue' for c in counts]
        ax.bar(counts, tc, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(pref_count, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'Pref={pref_count}')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('Ball Count', fontsize=9)
        ax.set_ylabel('Mean Activation', fontsize=9)
        ax.set_title(f'Unit {unit_idx} | Sel={selectivity[unit_idx]:.3f} | Pref={pref_count}', 
                    fontsize=10, fontweight='bold')
        ax.set_xticks(counts)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle(f'{layer_name}: Top {top_k} Number-Selective Units (Final Timestep)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    tuning_path = os.path.join(save_dir, f'{layer_name.lower()}_number_selective_units_tuning_curves.png')
    plt.savefig(tuning_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 调谐曲线已保存: {tuning_path}")
    
    # ===== Visualization 2: Selectivity distribution histogram =====
    print(f"生成选择性分布直方图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of all selectivity scores
    ax.hist(selectivity, bins=50, color='steelblue', alpha=0.7, edgecolor='black', label='All units')
    
    # Vertical line at 90th percentile
    ax.axvline(percentile_90, color='red', linestyle='--', linewidth=2.5, 
              label=f'90th percentile ({percentile_90:.3f})')
    
    # Highlight top units
    ax.scatter(selectivity[top_indices], np.ones(top_k)*5, color='orange', s=100, 
              marker='*', edgecolor='darkred', linewidth=1.5, label=f'Top {top_k} units', zorder=5)
    
    ax.set_xlabel('Selectivity Index (std/mean)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Units', fontsize=11, fontweight='bold')
    ax.set_title(f'{layer_name}: Selectivity Distribution\n{percent_selective:.1f}% of units above 90th percentile', 
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    dist_path = os.path.join(save_dir, f'{layer_name.lower()}_selectivity_distribution.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 选择性分布已保存: {dist_path}")
    
    # ===== Data Export =====
    print(f"导出数据到CSV...")
    
    # Export tuning curves
    tuning_csv = os.path.join(save_dir, f'{layer_name.lower()}_tuning_curves.csv')
    header = 'unit_idx,' + ','.join([f'count_{c}' for c in range(11)])
    data = np.column_stack([np.arange(H), tuning_curves])
    np.savetxt(tuning_csv, data, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f"✓ 调谐曲线已保存: {tuning_csv}")
    
    # Export selectivity scores
    selectivity_csv = os.path.join(save_dir, f'{layer_name.lower()}_selectivity_scores.csv')
    sel_data = np.column_stack([np.arange(H), selectivity])
    np.savetxt(selectivity_csv, sel_data, delimiter=',', header='unit_idx,selectivity', 
              comments='', fmt='%.6f')
    print(f"✓ 选择性得分已保存: {selectivity_csv}")
    
    # Export top units
    top_units_csv = os.path.join(save_dir, f'{layer_name.lower()}_top_{top_k}_units.csv')
    pref_counts = np.argmax(np.abs(tuning_curves[top_indices]), axis=1)
    top_data = np.column_stack([np.arange(top_k), top_indices, top_selectivity, pref_counts])
    np.savetxt(top_units_csv, top_data, delimiter=',', 
              header='rank,unit_idx,selectivity,preferred_count', comments='', fmt='%d,%.0f,%.6f,%.0f')
    print(f"✓ 前{top_k}个单元已保存: {top_units_csv}")
    
    result = {
        'tuning_curves': tuning_curves,
        'selectivity': selectivity,
        'top_units': top_indices.tolist(),
        'percent_selective': percent_selective,
    }
    
    return result


def plot_gates_over_time(gates: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # gates: [N, T, 2] -> avg over N
    avg = gates.mean(axis=0)  # [T, 2]
    T_len = avg.shape[0]
    x = np.arange(T_len)

    plt.figure(figsize=(8,4))
    plt.plot(x, avg[:,0], label='visual weight', color='royalblue')
    plt.plot(x, avg[:,1], label='joint weight', color='orange')
    plt.ylim(0, 1)
    plt.xlabel('timestep'); plt.ylabel('gate weight')
    plt.title('Average Modality Gate Weights over Time')
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gate_weights_over_time.png'), dpi=200)
    plt.close()


def visualize_softmax_summary(logits_last: np.ndarray, labels: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    probs = F.softmax(torch.from_numpy(logits_last).float(), dim=1).numpy()
    num_classes = probs.shape[1]

    # mean per class
    mean_probs = probs.mean(axis=0)
    plt.figure(figsize=(8,4))
    plt.bar(range(num_classes), mean_probs, color='steelblue')
    plt.xlabel('Class'); plt.ylabel('Mean confidence'); plt.title('Mean Softmax Confidence (last step)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'softmax_summary.png'), dpi=200)
    plt.close()


class SequenceGradCAM:
    def __init__(self, model: SimplifiedEmbodiedCountingModel, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        self.grad_magnitudes = []  # For diagnostics

        def fwd_hook(m, inp, out):
            # Save activation and register hook for gradients
            out_clone = out.clone()
            out_clone.requires_grad_(True)
            out_clone.retain_grad()
            self.activations.append(out_clone)
            
            def grad_hook(grad):
                self.gradients.append(grad.clone())
                # Compute gradient magnitude for diagnostics
                grad_mag = grad.abs().mean().item()
                self.grad_magnitudes.append(grad_mag)
            
            if out_clone.requires_grad:
                out_clone.register_hook(grad_hook)
            
            return out_clone

        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)

    def remove(self):
        self._fwd_handle.remove()

    def generate_cam_for_t(self, sequence_data: dict, t_index: int, target_class: int | None, device: torch.device, 
                          use_activation_weighting: bool = True):
        """Generate Grad-CAM for timestep t_index.
        
        Args:
            use_activation_weighting: If True, weight gradients by activation magnitude
                                    to help with vanishing gradients in early timesteps.
        """
        self.activations.clear()
        self.gradients.clear()
        self.grad_magnitudes.clear()

        # 1) Get target class if not provided (eval pass without grad)
        self.model.eval()
        with torch.no_grad():
            out_eval = self.model({
                'images': sequence_data['images'].unsqueeze(0).to(device),
                'joints': sequence_data['joints'].unsqueeze(0).to(device),
            })
            counts = out_eval['counts']  # [1, T, C]
            if target_class is None:
                target_class = counts[0, t_index].argmax().item()

        # 2) Backprop pass with grad - keep in eval mode but enable gradients
        self.model.eval()  # Keep in eval mode to avoid dropout/batchnorm changes
        images = sequence_data['images'].unsqueeze(0).to(device).clone().requires_grad_(True)
        joints = sequence_data['joints'].unsqueeze(0).to(device)
        
        out = self.model({'images': images, 'joints': joints})
        score = out['counts'][0, t_index, target_class]
        
        # Zero gradients before backward
        if images.grad is not None:
            images.grad.zero_()
        
        score.backward()

        # Select the activation/grad at timestep t_index
        # Assume visual encoder called once per timestep in order
        assert len(self.activations) > t_index and len(self.gradients) > t_index, (
            f"Hooks captured {len(self.activations)} steps, need {t_index+1}")
        act = self.activations[t_index]       # [B, C, H, W] already cloned
        grad = self.gradients[t_index]        # already cloned
        # Use batch 0
        act = act[0]  # [C, H, W]
        grad = grad[0]  # [C, H, W]

        # Improved weighting strategy for vanishing gradient case
        if use_activation_weighting:
            # Weight by both gradient AND activation magnitude
            # This helps recover CAM signal in early timesteps with weak gradients
            act_weights = act.abs().mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            grad_weights = grad.abs().mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            # Combine: use activation magnitude if gradient is too weak
            combined_weights = grad_weights + 0.1 * act_weights  # Bias towards gradients but use activations as fallback
            cam = (combined_weights * act).sum(dim=0, keepdim=False)  # [H, W]
        else:
            # Standard Grad-CAM
            weights = grad.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
            cam = (weights * act).sum(dim=0, keepdim=False)  # [H,W]
        
        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        return cam_np, target_class
    
    def get_gradient_diagnostics(self) -> dict:
        """Return diagnostic info about gradient magnitudes across timesteps."""
        return {
            'num_activations': len(self.activations),
            'grad_magnitudes': self.grad_magnitudes.copy(),
            'mean_grad': np.mean(self.grad_magnitudes) if self.grad_magnitudes else 0,
        }


def visualize_sequence_gradcam(model: SimplifiedEmbodiedCountingModel,
                               dataset,
                               device: torch.device,
                               logits_btC: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               samples_per_class: int = 2,
                               timestep_policy: str = 'last',
                               labels: np.ndarray = None):
    import math
    from torch.utils.data import DataLoader

    # Determine per-class indices using true labels (not predictions)
    num_classes = logits_btC.shape[2]
    per_class = {c: [] for c in range(num_classes)}
    sample_labels = labels if labels is not None else preds  # Fallback to preds if labels not provided
    for i, label in enumerate(sample_labels):
        if len(per_class[label]) < samples_per_class:
            per_class[label].append(i)

    selected = []
    for c in range(num_classes):
        selected.extend(per_class[c])
    if not selected:
        print("⚠ 无可视化样本（按类别采样为空）")
        return

    # Small loader with batch_size=1 for deterministic indexing
    use_pin = device.type == 'cuda'
    small_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_pin)

    # Build map from global index to sequence data
    seq_map = {}
    idx_counter = 0
    for batch in small_loader:
        if idx_counter in selected:
            # Remove batch dimension since batch_size=1
            seq_data = {
                'images': batch['sequence_data']['images'][0],  # [T,3,224,224]
                'joints': batch['sequence_data']['joints'][0],  # [T,2]
            }
            seq_map[idx_counter] = seq_data
            if len(seq_map) == len(selected):
                break
        idx_counter += 1

    # Grad-CAM setup - multiple layers
    target_layers = {
        'conv1': model.visual_encoder.features[0],  # First conv layer
        'conv3': model.visual_encoder.features[6],  # Third conv layer
        'conv5': model.visual_encoder.features[-3], # Fifth conv layer (last)
    }
    cam_engines = {name: SequenceGradCAM(model, layer) for name, layer in target_layers.items()}

    # Figure layout: 2 samples per row, 4 cols per sample (Conv1 CAM + Conv3 CAM + Conv5 CAM + Bars)
    n = len(selected)
    samples_per_row = 2
    cols = samples_per_row * 4
    rows = math.ceil(n / samples_per_row)
    fig = plt.figure(figsize=(16, 5*rows))
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    for idx, gidx in enumerate(selected):
        seq = seq_map[gidx]
        images = seq['images']  # [T,3,224,224]
        T = images.shape[0]
        t_index = T-1 if timestep_policy == 'last' else T-1

        # Generate CAMs for all three layers
        cams = {}
        for layer_name, cam_engine in cam_engines.items():
            cam_np, tgt_cls = cam_engine.generate_cam_for_t(seq, t_index=t_index, target_class=preds[gidx].item() if hasattr(preds[gidx], 'item') else int(preds[gidx]), device=device, use_activation_weighting=True)
            cams[layer_name] = cam_np

        # De-normalize frame at t_index
        frame = images[t_index:t_index+1].to(device)
        frame_denorm = (frame * imagenet_std + imagenet_mean).squeeze(0).permute(1,2,0).detach().cpu().numpy()
        frame_denorm = np.clip(frame_denorm*255, 0, 255).astype(np.uint8)

        # Generate overlays for each layer
        overlays = {name: generate_gradcam_overlay(frame_denorm, cam) for name, cam in cams.items()}

        # Softmax bars from logits at t_index
        probs_full = torch.softmax(torch.from_numpy(logits_btC[gidx, t_index:t_index+1, :]).float(), dim=1).numpy()[0]
        # If 11 classes, display last 10 (1-10)
        if len(probs_full) > 10:
            probs = probs_full[-10:]
            label_offset = len(probs_full) - 10
        else:
            probs = probs_full
            label_offset = 0
        num_show = len(probs)
        adjusted_pred = int(preds[gidx]) - label_offset

        row = idx // samples_per_row
        cstart = (idx % samples_per_row) * 4
        ax_conv1 = fig.add_subplot(gs[row, cstart])
        ax_conv3 = fig.add_subplot(gs[row, cstart+1])
        ax_conv5 = fig.add_subplot(gs[row, cstart+2])
        ax_bar = fig.add_subplot(gs[row, cstart+3])

        # Titles
        true_label = int(dataset.csv_data.iloc[gidx]['ball_count']) if hasattr(dataset, 'csv_data') else None
        pred_label = int(preds[gidx])
        title = f"True: {true_label} | Pred: {pred_label}" if true_label is not None else f"Pred: {pred_label}"
        title_color = ('green' if true_label==pred_label else 'red') if true_label is not None else 'black'
        
        # Display Conv1 CAM
        ax_conv1.imshow(overlays['conv1'])
        ax_conv1.axis('off')
        ax_conv1.set_title(f"Conv1\n{title}", color=title_color, fontsize=10, fontweight='bold')
        
        # Display Conv3 CAM
        ax_conv3.imshow(overlays['conv3'])
        ax_conv3.axis('off')
        ax_conv3.set_title(f"Conv3\n{title}", color=title_color, fontsize=10, fontweight='bold')
        
        # Display Conv5 CAM
        ax_conv5.imshow(overlays['conv5'])
        ax_conv5.axis('off')
        ax_conv5.set_title(f"Conv5\n{title}", color=title_color, fontsize=10, fontweight='bold')

        class_labels = [str(i+1) for i in range(num_show)]
        colors = ['lightcoral' if i == adjusted_pred else 'steelblue' for i in range(num_show)]
        ax_bar.bar(range(num_show), probs, color=colors, alpha=0.85, edgecolor='black')
        ax_bar.set_xlabel('Ball Count'); ax_bar.set_ylabel('Probability')
        ax_bar.set_title('Softmax Probabilities')
        ax_bar.set_xticks(range(num_show))
        ax_bar.set_xticklabels(class_labels)
        ax_bar.set_ylim([0,1])
        ax_bar.grid(axis='y', alpha=0.3)
        if 0 <= adjusted_pred < num_show:
            ax_bar.text(adjusted_pred, probs[adjusted_pred]+0.02, f"{probs[adjusted_pred]:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    for cam_engine in cam_engines.values():
        cam_engine.remove()
    print(f"✓ Grad-CAM可视化已保存: {save_path}")


def visualize_sequence_gradcam_all_steps(model: SimplifiedEmbodiedCountingModel,
                                         dataset,
                                         device: torch.device,
                                         logits_btC: np.ndarray,
                                         preds: np.ndarray,
                                         save_dir: str,
                                         samples_per_class: int = 1,
                                         labels: np.ndarray = None):
    """For selected samples, save an image per sample showing CAM overlays for all timesteps.
    Files are saved as sample_{global_index}_all_steps.png under save_dir.
    """
    import math
    from torch.utils.data import DataLoader

    os.makedirs(save_dir, exist_ok=True)

    num_classes = logits_btC.shape[2]
    per_class = {c: [] for c in range(num_classes)}
    sample_labels = labels if labels is not None else preds  # Fallback to preds if labels not provided
    for i, label in enumerate(sample_labels):
        if len(per_class[label]) < samples_per_class:
            per_class[label].append(i)

    selected = []
    for c in range(num_classes):
        selected.extend(per_class[c])
    if not selected:
        print("⚠ 无可视化样本（按类别采样为空）")
        return

    use_pin = device.type == 'cuda'
    small_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_pin)

    seq_map = {}
    idx_counter = 0
    for batch in small_loader:
        if idx_counter in selected:
            seq_data = {
                'images': batch['sequence_data']['images'][0],
                'joints': batch['sequence_data']['joints'][0],
            }
            seq_map[idx_counter] = seq_data
            if len(seq_map) == len(selected):
                break
        idx_counter += 1

    target_layers = {
        'conv1': model.visual_encoder.features[0],
        'conv3': model.visual_encoder.features[6],
        'conv5': model.visual_encoder.features[-3],
    }
    cam_engines = {name: SequenceGradCAM(model, layer) for name, layer in target_layers.items()}

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    for gidx in selected:
        seq = seq_map[gidx]
        images = seq['images']  # [T,3,224,224]
        T = images.shape[0]

        # Prepare overlays for all timesteps and all layers
        overlays_per_layer = {layer_name: [] for layer_name in target_layers.keys()}
        titles = []
        per_timestep_probs = []
        for t in range(T):
            # CAM for step t with improved weighting - for each layer
            target_cls = preds[gidx].item() if hasattr(preds[gidx], 'item') else int(preds[gidx])
            for layer_name, cam_engine in cam_engines.items():
                cam_np, _ = cam_engine.generate_cam_for_t(seq, t_index=t, target_class=target_cls, device=device, use_activation_weighting=True)
                
                # De-normalize frame at t
                frame = images[t:t+1].to(device)
                frame_denorm = (frame * imagenet_std + imagenet_mean).squeeze(0).permute(1,2,0).detach().cpu().numpy()
                frame_denorm = np.clip(frame_denorm*255, 0, 255).astype(np.uint8)

                overlay = generate_gradcam_overlay(frame_denorm, cam_np)
                overlays_per_layer[layer_name].append(overlay)

            # Per-step predicted label (from logits)
            probs = torch.softmax(torch.from_numpy(logits_btC[gidx, t:t+1, :]).float(), dim=1).numpy()[0]
            per_timestep_probs.append(probs)
            if len(probs) > 10:
                # if includes 0-class, use last 10 as 1-10
                pred_t = probs[-10:].argmax() + (len(probs) - 10)
            else:
                pred_t = probs.argmax()
            titles.append(f"t={t} pred={pred_t}")

        # Save separate image for each layer's all-steps visualization
        for layer_name, overlays in overlays_per_layer.items():
            fig = plt.figure(figsize=(3*T, 3))
            gs = fig.add_gridspec(1, T, wspace=0.1)
            for t in range(T):
                ax = fig.add_subplot(gs[0, t])
                ax.imshow(overlays[t])
                ax.axis('off')
                ax.set_title(titles[t], fontsize=9)
            out_path = os.path.join(save_dir, f"sample_{gidx}_{layer_name}_all_steps.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Saved CAM ({layer_name}) for sample {gidx}")

        # Extra figure: per-timestep softmax bar charts
        probs_arr = np.stack(per_timestep_probs, axis=0)  # [T, C]
        # If there is a dummy 0-class, also plot only last 10 entries for readability
        fig = plt.figure(figsize=(3*T, 3))
        gs = fig.add_gridspec(1, T, wspace=0.2)
        for t in range(T):
            ax = fig.add_subplot(gs[0, t])
            probs_t = probs_arr[t]
            if len(probs_t) > 10:
                probs_plot = probs_t[-10:]
                label_offset = len(probs_t) - 10
                class_labels = [str(i) for i in range(1, 11)]  # Always 1-10
            else:
                probs_plot = probs_t
                class_labels = [str(i+1) for i in range(len(probs_t))]
                label_offset = 0
            pred_idx = probs_plot.argmax()
            colors = ['lightcoral' if i == pred_idx else 'steelblue' for i in range(len(probs_plot))]
            ax.bar(range(len(probs_plot)), probs_plot, color=colors, alpha=0.85, edgecolor='black')
            ax.set_ylim([0,1])
            ax.set_xticks(range(len(probs_plot)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(f"t={t}")
            if 0 <= pred_idx < len(probs_plot):
                ax.text(pred_idx, probs_plot[pred_idx]+0.02, f"{probs_plot[pred_idx]:.2f}", ha='center', va='bottom', fontsize=7, fontweight='bold', color='red')
        fig.suptitle(f"Sample {gidx} Softmax per timestep", fontsize=12)
        plt.tight_layout()
        bar_out = os.path.join(save_dir, f"sample_{gidx}_all_steps_bars.png")
        plt.savefig(bar_out, dpi=150)
        plt.close()

        # Save softmax data to CSV
        softmax_csv_out = os.path.join(save_dir, f"sample_{gidx}_softmax.csv")
        softmax_rows = []
        for t in range(T):
            probs_t = probs_arr[t]
            if len(probs_t) > 10:
                # Only save last 10 (classes 1-10)
                for class_id in range(1, 11):
                    prob_val = probs_t[class_id] if class_id < len(probs_t) else 0.0
                    softmax_rows.append([gidx, t, class_id, prob_val])
            else:
                # Save all classes
                for class_id in range(len(probs_t)):
                    softmax_rows.append([gidx, t, class_id+1, probs_t[class_id]])
        softmax_data = np.array(softmax_rows)
        np.savetxt(softmax_csv_out, softmax_data, delimiter=",", header="sample_id,timestep,ball_count,probability", comments="", fmt="%.6f")


    for cam_engine in cam_engines.values():
        cam_engine.remove()
    print(f"✓ Grad-CAM全时间步可视化已保存到: {save_dir}")


def run_analysis(args):
    # device
    if args.device.lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device.lower())
    print(f"使用设备: {device}")

    # Extract epoch number from checkpoint path
    import re
    epoch_num = None
    ckpt_name = os.path.basename(args.checkpoint)
    match = re.search(r'epoch_(\d+)', ckpt_name)
    if match:
        epoch_num = match.group(1)
    
    # output dirs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if epoch_num:
        out_root = os.path.join(args.output_dir, f'embodied_analysis_epoch{epoch_num}_{ts}')
    else:
        out_root = os.path.join(args.output_dir, f'embodied_analysis_{ts}')
    feat_dir = os.path.join(out_root, 'features')
    pred_dir = os.path.join(out_root, 'predictions')
    gate_dir = os.path.join(out_root, 'gates')
    attn_dir = os.path.join(out_root, 'attention_maps')
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gate_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)

    # load model
    model, cfg = load_embodied_checkpoint(args.checkpoint, device)

    # data
    train_loader, val_loader = get_ball_counting_data_loaders(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        normalize_images=True,
        shuffle_joints=False,
        curriculum_mode='random',
    )

    # pin_memory adjustment for cpu
    if device.type != 'cuda':
        val_loader = DataLoader(val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # extract
    logits_btC, labels, preds, hidden_layer1, hidden_layer2, gates_bt2 = extract_sequence_outputs(model, val_loader, device)

    # confusion & per-class
    from visualization_utils import plot_per_class_accuracy
    plot_confusion_matrix(labels, preds, os.path.join(pred_dir, 'confusion_matrix.png'), normalize=False, class_names=[str(i) for i in range(1,11)])
    plot_confusion_matrix(labels, preds, os.path.join(pred_dir, 'confusion_matrix_normalized.png'), normalize=True, class_names=[str(i) for i in range(1,11)])
    plot_per_class_accuracy(labels, preds, os.path.join(pred_dir, 'per_class_accuracy.png'))

    # softmax summary (last step)
    visualize_softmax_summary(logits_btC[:, -1, :], labels, pred_dir)

    # LSTM PCA (scatter + trajectories) - BOTH LAYERS, 2D+3D
    if hidden_layer1 is not None:
        # Layer 1 (top layer)
        layer1_dir = os.path.join(feat_dir, 'layer1')
        os.makedirs(layer1_dir, exist_ok=True)
        pca2d_l1, pca3d_l1, _, _, _, _ = plot_lstm_pca(hidden_layer1, labels, layer1_dir, layer_name="Layer1")
        plot_lstm_trajectories(hidden_layer1, labels, layer1_dir, pca2d_model=pca2d_l1, pca3d_model=pca3d_l1, layer_name="Layer1")
        
        # Number selectivity analysis for Layer1
        try:
            num_select_dir = os.path.join(feat_dir, 'layer1', 'number_selectivity')
            analyze_number_selectivity(hidden_layer1, labels, num_select_dir, layer_name="Layer1", top_k=50)
        except Exception as e:
            print(f"⚠ Layer1数字选择性分析失败: {e}")
        
        # Number line RSA for Layer1
        try:
            rsa_dir = os.path.join(feat_dir, 'layer1', 'number_line_rsa')
            compute_number_line_rsa(hidden_layer1, labels, rsa_dir, layer_name="Layer1")
        except Exception as e:
            print(f"⚠ Layer1数字线RSA分析失败: {e}")
        
        # Rotational dynamics for Layer1
        try:
            rotation_dir = os.path.join(feat_dir, 'layer1', 'rotational_dynamics')
            analyze_rotational_dynamics(hidden_layer1, labels, pca2d_l1, rotation_dir, layer_name="Layer1")
        except Exception as e:
            print(f"⚠ Layer1旋转动力学分析失败: {e}")
        
        # jPCA analysis for Layer1
        try:
            jpca_dir = os.path.join(feat_dir, 'layer1', 'jPCA')
            jpca_results_l1 = perform_jPCA_analysis(
                hidden_btH=hidden_layer1,
                labels=labels,
                save_dir=jpca_dir,
                layer_name="Layer1",
                pca_n_components=6
            )
            print(f"✓ Layer1 jPCA: ω={jpca_results_l1['rotation_frequency_omega']:.3f}, "
                  f"rotation%={jpca_results_l1['rotation_variance_fraction']:.1%}")
        except Exception as e:
            print(f"⚠ Layer1 jPCA分析失败: {e}")
        
    if hidden_layer2 is not None:
        # Layer 2 (bottom layer)
        layer2_dir = os.path.join(feat_dir, 'layer2')
        os.makedirs(layer2_dir, exist_ok=True)
        pca2d_l2, pca3d_l2, _, _, _, _ = plot_lstm_pca(hidden_layer2, labels, layer2_dir, layer_name="Layer2")
        plot_lstm_trajectories(hidden_layer2, labels, layer2_dir, pca2d_model=pca2d_l2, pca3d_model=pca3d_l2, layer_name="Layer2")
        
        # Number selectivity analysis for Layer2
        try:
            num_select_dir = os.path.join(feat_dir, 'layer2', 'number_selectivity')
            analyze_number_selectivity(hidden_layer2, labels, num_select_dir, layer_name="Layer2", top_k=50)
        except Exception as e:
            print(f"⚠ Layer2数字选择性分析失败: {e}")
        
        # Number line RSA for Layer2
        try:
            rsa_dir = os.path.join(feat_dir, 'layer2', 'number_line_rsa')
            compute_number_line_rsa(hidden_layer2, labels, rsa_dir, layer_name="Layer2")
        except Exception as e:
            print(f"⚠ Layer2数字线RSA分析失败: {e}")
        
        # Rotational dynamics for Layer2
        try:
            rotation_dir = os.path.join(feat_dir, 'layer2', 'rotational_dynamics')
            analyze_rotational_dynamics(hidden_layer2, labels, pca2d_l2, rotation_dir, layer_name="Layer2")
        except Exception as e:
            print(f"⚠ Layer2旋转动力学分析失败: {e}")
        
        # jPCA analysis for Layer2
        try:
            jpca_dir = os.path.join(feat_dir, 'layer2', 'jPCA')
            jpca_results_l2 = perform_jPCA_analysis(
                hidden_btH=hidden_layer2,
                labels=labels,
                save_dir=jpca_dir,
                layer_name="Layer2",
                pca_n_components=6
            )
            print(f"✓ Layer2 jPCA: ω={jpca_results_l2['rotation_frequency_omega']:.3f}, "
                  f"rotation%={jpca_results_l2['rotation_variance_fraction']:.1%}")
        except Exception as e:
            print(f"⚠ Layer2 jPCA分析失败: {e}")

    # Gates
    if gates_bt2 is not None:
        plot_gates_over_time(gates_bt2, gate_dir)

    # Visual head PCA/t-SNE
    try:
        visual_dir = os.path.join(feat_dir, 'visual_head')
        os.makedirs(visual_dir, exist_ok=True)
        visual_btD, _ = extract_visual_features(model, val_loader, device)
        if visual_btD is not None:
            plot_visual_pca_tsne(visual_btD, labels, visual_dir)
    except Exception as e:
        print(f"⚠ 视觉头PCA/t-SNE失败: {e}")

    # Visual head Grad-CAM
    try:
        samples_per_class = getattr(args, 'gradcam_samples_per_class', 2)
        cam_mode = getattr(args, 'cam_timestep', 'last')
        if cam_mode == 'all':
            all_steps_dir = os.path.join(attn_dir, 'gradcam_all_steps')
            visualize_sequence_gradcam_all_steps(
                model=model,
                dataset=val_loader.dataset,
                device=device,
                logits_btC=logits_btC,
                preds=preds,
                save_dir=all_steps_dir,
                samples_per_class=samples_per_class,
                labels=labels,
            )
        else:
            visualize_sequence_gradcam(
                model=model,
                dataset=val_loader.dataset,
                device=device,
                logits_btC=logits_btC,
                preds=preds,
                save_path=os.path.join(attn_dir, 'gradcam_samples.png'),
                samples_per_class=samples_per_class,
                timestep_policy=cam_mode,
                labels=labels,
            )
    except Exception as e:
        print(f"⚠ Grad-CAM可视化失败: {e}")

    print(f"\n全部完成。输出目录: {out_root}")


def build_argparser():
    p = argparse.ArgumentParser(description='Embodied model analysis (visual head + LSTM PCA)')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection')
    p.add_argument('--train_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv')
    p.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv')
    p.add_argument('--output_dir', type=str, default='analysis_results')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--sequence_length', type=int, default=11)
    p.add_argument('--device', type=str, default='cpu', choices=['auto','cpu','cuda'])
    p.add_argument('--cam_timestep', type=str, default='all', choices=['last','all','first','mid'])
    p.add_argument('--gradcam_samples_per_class', type=int, default=2, help="Number of samples to visualize per class for Grad-CAM")
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    run_analysis(args)

