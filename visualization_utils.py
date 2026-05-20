"""
Visualization utilities for model analysis.
Includes: PCA, t-SNE, Grad-CAM, confusion matrix, per-class accuracy, error samples, softmax analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import cv2
import os


# ============ Feature Visualisation ============

def plot_pca(features: np.ndarray, 
             labels: np.ndarray, 
             save_path: str,
             n_components: int = 2,
             title: str = "PCA Visualization"):
    """
    Plot PCA scatter plot of feature embeddings.
    
    Args:
        features: [N, D] feature matrix
        labels: [N] class labels
        save_path: path to save the output image
        n_components: number of PCA components (2 or 3)
        title: plot title
    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Explained variance per component
    explained_var = pca.explained_variance_ratio_
    
    # Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                            c=labels, cmap='tab10', s=20, alpha=0.7)
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
        ax.set_title(f'{title}\nTotal variance explained: {explained_var.sum():.2%}')
        plt.colorbar(scatter, label='Ball Count', ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2],
                            c=labels, cmap='tab10', s=20, alpha=0.7)
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_var[2]:.2%})')
        ax.set_title(f'{title}\nVariance: {explained_var.sum():.2%}')
        plt.colorbar(scatter, label='Ball Count', ax=ax, shrink=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved PCA {n_components}D plot: {save_path}")
    print(f"  Variance per component: {', '.join([f'PC{i+1}={v:.2%}' for i, v in enumerate(explained_var)])}")


def plot_tsne(features: np.ndarray, 
              labels: np.ndarray, 
              save_path: str,
              n_components: int = 2,
              perplexity: int = 30,
              title: str = "t-SNE Visualization"):
    """
    Plot t-SNE scatter plot of feature embeddings.
    
    Args:
        features: [N, D] feature matrix
        labels: [N] class labels
        save_path: path to save the output image
        n_components: number of t-SNE components (2 or 3)
        perplexity: t-SNE perplexity parameter
        title: plot title
    """
    print(f"Running t-SNE (n_components={n_components}, perplexity={perplexity})...")
    
    # Fit t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=42, n_iter=1000, verbose=0)
    features_tsne = tsne.fit_transform(features)
    
    # Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                            c=labels, cmap='tab10', s=20, alpha=0.7)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f'{title} (perplexity={perplexity})')
        plt.colorbar(scatter, label='Ball Count', ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2],
                            c=labels, cmap='tab10', s=20, alpha=0.7)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.set_title(f'{title} (perplexity={perplexity})')
        plt.colorbar(scatter, label='Ball Count', ax=ax, shrink=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved t-SNE {n_components}D plot: {save_path}")


# ============ Grad-CAM ============

class GradCAM:
    """Grad-CAM implementation for visualising CNN activations."""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: the classifier model
            target_layer: the convolutional layer to hook into
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Disable in-place ReLU to allow gradient flow
        self._disable_inplace_relu(self.model)
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def _disable_inplace_relu(self, module):
        """Recursively disable in-place ReLU operations."""
        for child in module.children():
            if isinstance(child, nn.ReLU):
                child.inplace = False
            else:
                self._disable_inplace_relu(child)
    
    def save_activation(self, module, input, output):
        """Forward hook — stores the layer activation."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook — stores the gradient of the layer output."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate a Grad-CAM heatmap.
        
        Args:
            input_image: [1, C, H, W] input tensor
            target_class: class index to visualise; uses predicted class if None
        
        Returns:
            cam: [H, W] normalised heatmap in [0, 1]
            pred_class: the class index used for backprop
        """
        # Determine target class with a no-grad forward pass
        self.model.eval()
        with torch.no_grad():
            output_pred = self.model(input_image)
            if target_class is None:
                target_class = output_pred.argmax(dim=1).item()
        
        # Enable gradients and run forward pass
        self.model.train()
        input_image.requires_grad_(True)
        
        output = self.model(input_image)
        
        # Backpropagate through the target class score
        self.model.zero_grad()
        target_loss = output[0, target_class]
        target_loss.backward()
        
        # Global average pooling of gradients (weights)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted sum of activations, then ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()  # [H, W]
        
        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Restore eval mode
        self.model.eval()
        
        return cam, target_class


def generate_gradcam_overlay(image: np.ndarray, 
                             cam: np.ndarray,
                             alpha: float = 0.5,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an image.
    
    Args:
        image: [H, W, 3] RGB image in uint8 (0-255)
        cam: [H, W] CAM heatmap in [0, 1]
        alpha: blending weight for the heatmap
        colormap: OpenCV colormap to apply to the CAM
    
    Returns:
        overlay: [H, W, 3] blended image
    """
    # Resize CAM to match image dimensions
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, cam_colored, alpha, 0)
    
    return overlay


def visualize_gradcam_samples(model,
                              images: torch.Tensor,
                              labels: torch.Tensor,
                              predictions: torch.Tensor,
                              device: torch.device,
                              save_dir: str,
                              logits: np.ndarray = None,
                              samples_per_class: int = 2,
                              image_mean: List[float] = [0.485, 0.456, 0.406],
                              image_std: List[float] = [0.229, 0.224, 0.225]):
    """
    Generate Grad-CAM overlays for selected samples, with optional softmax bar charts.
    
    Args:
        model: the classifier model
        images: [N, C, H, W] input images
        labels: [N] ground-truth labels
        predictions: [N] predicted labels
        device: compute device
        save_dir: directory to save output images
        logits: [N, num_classes] raw logits; if provided, softmax bars are included
        samples_per_class: number of samples to visualise per class
        image_mean/std: ImageNet normalisation parameters for de-normalisation
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples — up to samples_per_class per class
    unique_labels = torch.unique(labels)
    selected_indices = []
    
    for cls in unique_labels:
        cls_mask = (labels == cls)
        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
        
        if len(cls_indices) == 0:
            continue
        
        # Sample at most samples_per_class indices from this class
        n_to_sample = min(samples_per_class, len(cls_indices))
        sampled = cls_indices[torch.randperm(len(cls_indices))[:n_to_sample]]
        selected_indices.extend(sampled.tolist())
    
    selected_indices = torch.tensor(selected_indices, dtype=torch.long)
    
    # Hook into Conv1, Conv3, Conv5 of the AlexNet feature extractor
    target_layers = {
        'conv1': model.visual_encoder.features[0],   # First conv layer
        'conv3': model.visual_encoder.features[6],   # Third conv layer
        'conv5': model.visual_encoder.features[-3],  # Fifth conv layer (last)
    }
    grad_cams = {name: GradCAM(model, layer) for name, layer in target_layers.items()}
    
    # De-normalisation constants
    mean = torch.tensor(image_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(image_std).view(1, 3, 1, 1).to(device)
    
    # Build figure layout:
    # With logits: 2 samples per row, 4 cols each (Conv1 CAM | Conv3 CAM | Conv5 CAM | Softmax bar)
    # Without logits: grid of Conv5 CAMs only
    n_samples = len(selected_indices)
    if logits is not None:
        n_samples_per_row = 2
        n_cols = n_samples_per_row * 4
        n_rows = (n_samples + n_samples_per_row - 1) // n_samples_per_row
        fig = plt.figure(figsize=(16, 5*n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    else:
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, sample_idx in enumerate(selected_indices):
        # Prepare input
        input_img = images[sample_idx:sample_idx+1].to(device)
        true_label = labels[sample_idx].item()
        pred_label = predictions[sample_idx].item()
        
        # Generate CAMs for all three layers
        cams = {}
        for layer_name, grad_cam in grad_cams.items():
            cam, _ = grad_cam.generate_cam(input_img, target_class=pred_label)
            cams[layer_name] = cam
        
        # De-normalise image for display
        img_denorm = input_img * std + mean
        img_denorm = img_denorm.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_denorm = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)
        
        # Overlay CAMs on the de-normalised image
        overlays = {name: generate_gradcam_overlay(img_denorm, cam) for name, cam in cams.items()}
        
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        title_text = f'True: {true_label} | Pred: {pred_label}'
        
        if logits is not None and sample_idx < len(logits):
            # Compute softmax probabilities
            probs_full = torch.softmax(torch.from_numpy(logits[sample_idx:sample_idx+1]).float(), dim=1).numpy()[0]
            
            # If there are 11 classes (0-10), display only the last 10 (classes 1-10)
            if len(probs_full) > 10:
                probs = probs_full[-10:]
                label_offset = len(probs_full) - 10
            else:
                probs = probs_full
                label_offset = 0
            
            num_classes = len(probs)
            
            # Place Conv1, Conv3, Conv5, and softmax bar in 4 columns per sample
            row = idx // 2
            col_start = (idx % 2) * 4
            ax_conv1 = fig.add_subplot(gs[row, col_start])
            ax_conv3 = fig.add_subplot(gs[row, col_start + 1])
            ax_conv5 = fig.add_subplot(gs[row, col_start + 2])
            ax_bar = fig.add_subplot(gs[row, col_start + 3])
            
            # Conv1 CAM
            ax_conv1.imshow(overlays['conv1'])
            ax_conv1.axis('off')
            ax_conv1.set_title(f'Conv1\n{title_text}', color=color, fontsize=10, fontweight='bold')
            
            # Conv3 CAM
            ax_conv3.imshow(overlays['conv3'])
            ax_conv3.axis('off')
            ax_conv3.set_title(f'Conv3\n{title_text}', color=color, fontsize=10, fontweight='bold')
            
            # Conv5 CAM
            ax_conv5.imshow(overlays['conv5'])
            ax_conv5.axis('off')
            ax_conv5.set_title(f'Conv5\n{title_text}', color=color, fontsize=10, fontweight='bold')
            
            # Softmax bar chart — class labels 1-10
            class_labels = [str(i+1) for i in range(num_classes)]
            adjusted_pred_label = pred_label - label_offset
            colors_bar = ['lightcoral' if i == adjusted_pred_label else 'steelblue' for i in range(num_classes)]
            ax_bar.bar(range(num_classes), probs, color=colors_bar, alpha=0.85, edgecolor='black')
            ax_bar.set_xlabel('Ball Count', fontsize=9)
            ax_bar.set_title('Softmax Probabilities', fontsize=10, fontweight='bold')
            ax_bar.set_xticks(range(num_classes))
            ax_bar.set_xticklabels(class_labels, fontsize=8)
            ax_bar.set_ylim([0, 1])
            ax_bar.grid(axis='y', alpha=0.3)
            
            # Annotate the predicted class bar with its probability
            if 0 <= adjusted_pred_label < num_classes:
                ax_bar.text(adjusted_pred_label, probs[adjusted_pred_label] + 0.02, 
                           f'{probs[adjusted_pred_label]:.2f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
        else:
            # No logits — show Conv5 CAM only
            axes[idx].imshow(overlays['conv5'])
            axes[idx].axis('off')
            axes[idx].set_title(title_text, color=color, fontsize=10, fontweight='bold')
    
    # Hide unused axes when logits are not provided
    if logits is None:
        for idx in range(len(selected_indices), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_samples.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Grad-CAM visualisation: {save_path}")
    print(f"  Samples: {len(selected_indices)}, per class: {samples_per_class}")
    print(f"  CAM layers: Conv1, Conv3, Conv5")


# ============ Prediction Analysis ============

def plot_confusion_matrix(labels: np.ndarray,
                         predictions: np.ndarray,
                         save_path: str,
                         class_names: List[str] = None,
                         normalize: bool = False):
    """
    Plot a confusion matrix heatmap.
    
    Args:
        labels: [N] ground-truth labels
        predictions: [N] predicted labels
        save_path: path to save the output image
        class_names: list of class name strings for axis labels
        normalize: if True, normalise counts by row (true class totals)
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix: {save_path}")


def plot_per_class_accuracy(labels: np.ndarray,
                           predictions: np.ndarray,
                           save_path: str,
                           class_names: List[str] = None):
    """
    Plot a per-class accuracy bar chart.
    
    Args:
        labels: [N] ground-truth labels
        predictions: [N] predicted labels
        save_path: path to save the output image
        class_names: list of class name strings for x-axis labels
    """
    # Compute per-class accuracy
    unique_labels = np.unique(labels)
    accuracies = []
    counts = []
    
    for label in unique_labels:
        mask = (labels == label)
        correct = (predictions[mask] == label).sum()
        total = mask.sum()
        acc = correct / total if total > 0 else 0
        accuracies.append(acc)
        counts.append(total)
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(unique_labels))
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.8)
    
    # Annotate each bar with accuracy and sample count
    for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}\n(n={count})',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names if class_names else unique_labels)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
               label=f'Mean Accuracy: {np.mean(accuracies):.1%}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved per-class accuracy plot: {save_path}")
    print(f"  Mean accuracy: {np.mean(accuracies):.2%}")


def visualize_error_samples(images: torch.Tensor,
                           labels: torch.Tensor,
                           predictions: torch.Tensor,
                           save_path: str,
                           n_samples: int = 20,
                           image_mean: List[float] = [0.485, 0.456, 0.406],
                           image_std: List[float] = [0.229, 0.224, 0.225]):
    """
    Visualise a random selection of misclassified samples.
    
    Args:
        images: [N, C, H, W] input images (normalised)
        labels: [N] ground-truth labels
        predictions: [N] predicted labels
        save_path: path to save the output image
        n_samples: maximum number of error samples to display
        image_mean/std: ImageNet normalisation parameters for de-normalisation
    """
    # Find all misclassified samples
    wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]
    
    if len(wrong_indices) == 0:
        print("No errors found — all predictions are correct.")
        return
    
    # Randomly select up to n_samples errors
    n_samples = min(n_samples, len(wrong_indices))
    selected = wrong_indices[torch.randperm(len(wrong_indices))[:n_samples]]
    
    # De-normalisation constants
    mean = torch.tensor(image_mean).view(1, 3, 1, 1)
    std = torch.tensor(image_std).view(1, 3, 1, 1)
    
    # Build grid layout
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, sample_idx in enumerate(selected):
        # De-normalise image
        img = images[sample_idx:sample_idx+1] * std + mean
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        true_label = labels[sample_idx].item()
        pred_label = predictions[sample_idx].item()
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                          color='red', fontsize=9, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Error Samples (Total errors: {len(wrong_indices)})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error samples: {save_path}")
    print(f"  Total errors: {len(wrong_indices)}, displayed: {n_samples}")


def visualize_softmax_outputs(logits: np.ndarray,
                              labels: np.ndarray,
                              predictions: np.ndarray,
                              save_path: str,
                              class_names: List[str] = None):
    """
    Visualise softmax confidence distributions across three panels:
    mean confidence per class, confidence histogram, correct vs wrong violin plot.
    
    Args:
        logits: [N, num_classes] raw logits
        labels: [N] ground-truth labels
        predictions: [N] predicted labels
        save_path: path to save the output image
        class_names: list of class name strings for axis labels
    """
    # Convert logits to softmax probabilities
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    
    num_classes = probs.shape[1]
    
    # Three-panel figure: mean confidence | confidence histogram | correct vs wrong
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Mean softmax confidence per class
    mean_probs = probs.mean(axis=0)
    axes[0].bar(range(num_classes), mean_probs, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Confidence', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Softmax Confidence per Class', fontsize=12, fontweight='bold')
    
    # Build tick labels — pad or trim to match num_classes
    tick_labels = class_names if class_names else [str(i) for i in range(num_classes)]
    if len(tick_labels) < num_classes:
        tick_labels = tick_labels + [str(i) for i in range(len(tick_labels), num_classes)]
    else:
        tick_labels = tick_labels[:num_classes]
    
    axes[0].set_xticks(range(num_classes), labels=tick_labels, rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Distribution of per-sample maximum confidence
    max_probs = probs.max(axis=1)
    axes[1].hist(max_probs, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {max_probs.mean():.3f}')
    axes[1].set_xlabel('Max Softmax Probability', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Maximum Softmax Probability', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Panel 3: Violin plot — confidence of correct vs wrong predictions
    correct_mask = (predictions == labels)
    correct_probs = max_probs[correct_mask]
    wrong_probs = max_probs[~correct_mask]
    
    data_to_plot = [correct_probs, wrong_probs]
    parts = axes[2].violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Correct', 'Wrong'], fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Max Softmax Probability', fontsize=11, fontweight='bold')
    axes[2].set_title('Confidence: Correct vs Wrong Predictions', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    # Annotate with sample count and mean
    axes[2].text(0, correct_probs.min() - 0.05, f'n={len(correct_probs)}\nmean={correct_probs.mean():.3f}',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[2].text(1, wrong_probs.min() - 0.05, f'n={len(wrong_probs)}\nmean={wrong_probs.mean():.3f}',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved softmax visualisation: {save_path}")
    print(f"  Mean confidence (correct): {correct_probs.mean():.4f}")
    print(f"  Mean confidence (wrong): {wrong_probs.mean():.4f}")


def save_analysis_summary(results: Dict, save_path: str):
    """
    Write a plain-text analysis summary file.
    
    Args:
        results: dict containing model info and performance metrics
        save_path: path to save the summary text file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Model information
        if 'model_info' in results:
            f.write("Model Information\n")
            for key, val in results['model_info'].items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
        
        # Overall accuracy metrics
        if 'overall_accuracy' in results:
            f.write("Overall Performance\n")
            f.write(f"  Accuracy: {results['overall_accuracy']:.2%}\n")
            f.write(f"  Total samples: {results['total_samples']}\n")
            f.write(f"  Correct: {results['correct_predictions']}\n")
            f.write(f"  Wrong: {results['wrong_predictions']}\n")
            f.write("\n")
        
        # Per-class accuracy
        if 'per_class_acc' in results:
            f.write("Per-Class Accuracy\n")
            for cls, acc in results['per_class_acc'].items():
                f.write(f"  Class {cls}: {acc:.2%}\n")
            f.write("\n")
        
        # PCA variance explained
        if 'pca_variance' in results:
            f.write("PCA\n")
            f.write(f"  Variance explained (3 PCs): {results['pca_variance']:.2%}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("End of summary.\n")
    
    print(f"Saved analysis summary: {save_path}")