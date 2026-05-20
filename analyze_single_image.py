"""
Single-image classifier analysis script.
Outputs: PCA/t-SNE feature visualisations and Grad-CAM attention maps.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add module search paths
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, 'Models'))
sys.path.append(os.path.join(CUR_DIR, 'Data_loader'))

from Single_Image_Classifier import create_single_image_model
from DataLoader_single_image import get_single_image_data_loaders
from visualization_utils import (
    plot_pca, plot_tsne, 
    visualize_gradcam_samples,
    plot_confusion_matrix, 
    plot_per_class_accuracy,
    visualize_error_samples,
    visualize_softmax_outputs,
    save_analysis_summary
)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load a model checkpoint and reconstruct the model.
    
    Args:
        checkpoint_path: path to the checkpoint file
        device: compute device to load the model onto
    
    Returns:
        model: loaded model in eval mode
        checkpoint: raw checkpoint dict
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Read config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config.get('num_classes', 11)
        use_pretrain = config.get('use_pretrain', True)
        print(f"  Checkpoint config:")
        print(f"    num_classes: {num_classes}")
        print(f"    use_pretrain: {use_pretrain}")
        if 'epoch' in checkpoint:
            print(f"    epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"    val_loss: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"    val_acc: {checkpoint['val_acc']:.2%}")
    else:
        # Fall back to defaults if no config is stored
        num_classes = 11
        use_pretrain = True
        print(f"  No config found in checkpoint — using defaults: num_classes={num_classes}, use_pretrain={use_pretrain}")
    
    # Instantiate model (weights will be loaded from checkpoint, not pretrained)
    model = create_single_image_model(
        num_classes=num_classes,
        use_pretrain=False,  # weights come from the checkpoint
        input_channels=3
    )
    
    # Load state dict — try several common key names
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint itself is the state dict; filter to matching keys
        filtered_state = {k: v for k, v in checkpoint.items() 
                         if k in model.state_dict()}
        if filtered_state:
            model.load_state_dict(filtered_state)
        else:
            raise ValueError(f"Unable to load model state. Checkpoint keys: {list(checkpoint.keys())}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully and set to eval mode.")
    print(f"{'='*60}\n")
    
    return model, checkpoint


def extract_features_and_predictions(model, dataloader, device):
    """
    Extract visual features, logits, and predictions for an entire dataloader.
    
    Args:
        model: model to run inference with
        dataloader: DataLoader to iterate over
        device: compute device
    
    Returns:
        features: [N, D] visual encoder features
        labels: [N] ground-truth labels
        predictions: [N] predicted class indices
        images: [N, C, H, W] input images (on CPU)
        logits: [N, num_classes] raw classification logits
    """
    print(f"\n{'='*60}")
    print("Extracting features and predictions...")
    print(f"{'='*60}")
    
    all_features = []
    all_labels = []
    all_predictions = []
    all_images = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Extract visual features from the AlexNet encoder
            features = model.visual_encoder(images)  # [B, 256]
            
            # Classify features
            logits = model.classifier(features)  # [B, num_classes]
            predictions = logits.argmax(dim=1)
            
            # Accumulate batch results
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_images.append(images.cpu())
            all_logits.append(logits.cpu())
    
    # Concatenate across batches
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    images = torch.cat(all_images, dim=0)
    logits = torch.cat(all_logits, dim=0).numpy()
    
    # Compute overall accuracy
    accuracy = (predictions == labels).mean()
    
    print(f"\nExtraction summary:")
    print(f"  Total samples: {len(features)}")
    print(f"  Feature dim: {features.shape[1]}")
    print(f"  Overall accuracy: {accuracy:.2%}")
    print(f"{'='*60}\n")
    
    return features, labels, predictions, images, logits


def run_analysis(args):
    """
    Run the full analysis pipeline.
    
    Args:
        args: parsed command-line arguments
    """
    # Set compute device
    if args.device.lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device.lower())
    print(f"Device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    feature_dir = os.path.join(output_dir, 'features')
    attention_dir = os.path.join(output_dir, 'attention_maps')
    prediction_dir = os.path.join(output_dir, 'predictions')
    
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}\n")
    
    # 1. Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # 2. Build data loaders
    print(f"{'='*60}")
    print("Building data loaders...")
    print(f"{'='*60}")
    
    train_loader, val_loader = get_single_image_data_loaders(
        train_csv_path=args.train_csv if args.train_csv else 'data/Tools_script/ball_counting_dataset_train_10.csv',
        val_csv_path=args.val_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=11,
        normalize_images=True
    )
    
    # Disable pin_memory on CPU (only beneficial for CUDA)
    use_pin_memory = device.type == 'cuda'
    
    # Rebuild val_loader without pin_memory when running on CPU
    if not use_pin_memory:
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False
        )
    
    # Use full validation set as the primary loader
    dataloader_full = val_loader
    print(f"Validation set size: {len(dataloader_full.dataset)}\n")
    
    # Extract features for PCA/t-SNE over the full validation set
    print("Extracting features for PCA/t-SNE visualisation...")
    features_full, labels_full, predictions_full, images_full, logits_full = extract_features_and_predictions(
        model, dataloader_full, device
    )
    
    # Optionally use a subset for Grad-CAM and error analysis
    if args.n_samples > 0:
        # Build a subset DataLoader
        from torch.utils.data import Subset
        indices = list(range(min(args.n_samples, len(dataloader_full.dataset))))
        subset = Subset(dataloader_full.dataset, indices)
        dataloader_subset = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory
        )
        print(f"Extracting subset features for Grad-CAM / error analysis...")
        features_subset, labels_subset, predictions_subset, images_subset, logits_subset = extract_features_and_predictions(
            model, dataloader_subset, device
        )
        print(f"Subset size: {len(dataloader_subset.dataset)}\n")
    else:
        # Use full set for all analyses
        features_subset = features_full
        labels_subset = labels_full
        predictions_subset = predictions_full
        images_subset = images_full
        logits_subset = logits_full
    
    # 4. Feature visualisation (PCA and t-SNE)
    print(f"{'='*60}")
    print("Generating feature visualisations (PCA and t-SNE)...")
    print(f"{'='*60}\n")
    
    # PCA 2D
    plot_pca(
        features_full, labels_full,
        save_path=os.path.join(feature_dir, 'pca_2d.png'),
        n_components=2,
        title="PCA 2D Visualization (All Validation Data)"
    )

    # Export PCA components to CSV (2D and 3D)
    try:
        from sklearn.decomposition import PCA
        # 2D PCA
        pca_2d = PCA(n_components=2)
        features_pca_2d = pca_2d.fit_transform(features_full)
        # Columns: [pc1, pc2, label, prediction]
        pca_2d_csv = np.column_stack([
            features_pca_2d,
            labels_full,
            predictions_full
        ])
        np.savetxt(
            os.path.join(feature_dir, 'pca_2d.csv'),
            pca_2d_csv,
            delimiter=',',
            header='pc1,pc2,label,prediction',
            comments=''
        )

        # 3D PCA (used later for variance calculation)
        pca_3d = PCA(n_components=3)
        features_pca_3d = pca_3d.fit_transform(features_full)
        pca_3d_csv = np.column_stack([
            features_pca_3d,
            labels_full,
            predictions_full
        ])
        np.savetxt(
            os.path.join(feature_dir, 'pca_3d.csv'),
            pca_3d_csv,
            delimiter=',',
            header='pc1,pc2,pc3,label,prediction',
            comments=''
        )
    except Exception as e:
        print(f"Failed to export PCA CSV: {e}")
    
    # t-SNE 2D (skip if dataset is too large)
    if len(features_full) <= 5000:
        plot_tsne(
            features_full, labels_full,
            save_path=os.path.join(feature_dir, 'tsne_2d.png'),
            n_components=2,
            perplexity=min(30, len(features_full) // 5),
            title="t-SNE 2D Visualization (All Validation Data)"
        )
    else:
        print(f"Dataset too large ({len(features_full)} samples) for t-SNE — skipping (threshold: 5000 samples)")
    
    print()
    
    # 5. Grad-CAM visualisation
    print(f"{'='*60}")
    print("Generating Grad-CAM attention maps...")
    print(f"{'='*60}\n")
    
    visualize_gradcam_samples(
        model=model,
        images=images_subset,
        labels=torch.from_numpy(labels_subset),
        predictions=torch.from_numpy(predictions_subset),
        device=device,
        save_dir=attention_dir,
        logits=logits_subset,
        samples_per_class=args.gradcam_samples_per_class
    )
    
    print()
    
    # 6. Prediction analysis
    print(f"{'='*60}")
    print("Generating prediction analysis plots...")
    print(f"{'='*60}\n")
    
    # Class names: 1–10
    class_names = [str(i) for i in range(1, 11)]
    
    # Raw confusion matrix
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix.png'),
        class_names=class_names,
        normalize=False
    )
    
    # Normalised confusion matrix
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix_normalized.png'),
        class_names=class_names,
        normalize=True
    )
    
    # Per-class accuracy bar chart
    plot_per_class_accuracy(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'per_class_accuracy.png'),
        class_names=class_names
    )
    
    # Error sample visualisation
    visualize_error_samples(
        images=images_subset,
        labels=torch.from_numpy(labels_subset),
        predictions=torch.from_numpy(predictions_subset),
        save_path=os.path.join(prediction_dir, 'error_samples.png'),
        n_samples=args.error_samples
    )
    
    # 7. Softmax output visualisation
    print(f"{'='*60}")
    print("Generating softmax output visualisation...")
    print(f"{'='*60}\n")
    
    visualize_softmax_outputs(
        logits=logits_subset,
        labels=labels_subset,
        predictions=predictions_subset,
        save_path=os.path.join(prediction_dir, 'softmax_outputs.png'),
        class_names=class_names
    )
    
    print()
    
    # 8. Save analysis summary
    print(f"{'='*60}")
    print("Saving analysis summary...")
    print(f"{'='*60}\n")
    
    # Compute per-class accuracy
    per_class_acc = {}
    unique_labels = np.unique(labels_full)
    for label in unique_labels:
        mask = (labels_full == label)
        acc = (predictions_full[mask] == label).mean()
        per_class_acc[int(label)] = acc
    
    # Compute PCA variance explained (use already-fitted pca_3d if available)
    from sklearn.decomposition import PCA
    try:
        pca_variance = pca_3d.explained_variance_ratio_.sum()
    except NameError:
        pca_tmp = PCA(n_components=3)
        pca_tmp.fit(features_full)
        pca_variance = pca_tmp.explained_variance_ratio_.sum()
    
    results = {
        'model_info': {
            'checkpoint': os.path.basename(args.checkpoint),
            'device': str(device),
            'num_classes': len(unique_labels),
            'total_validation_samples': len(labels_full),
            'analysis_samples': len(labels_subset) if args.n_samples > 0 else len(labels_full)
        },
        'overall_accuracy': (predictions_full == labels_full).mean(),
        'total_samples': len(labels_full),
        'correct_predictions': (predictions_full == labels_full).sum(),
        'wrong_predictions': (predictions_full != labels_full).sum(),
        'per_class_acc': per_class_acc,
        'pca_variance': pca_variance
    }
    
    save_analysis_summary(
        results,
        save_path=os.path.join(output_dir, 'analysis_summary.txt')
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    features/           # PCA and t-SNE visualisations")
    print(f"    attention_maps/     # Grad-CAM overlays")
    print(f"    predictions/        # Confusion matrix and accuracy plots")
    print(f"    analysis_summary.txt  # Overall results summary")
    print()


def main():
    parser = argparse.ArgumentParser(description='Single-image classifier analysis script')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint file')
    parser.add_argument('--val_csv', type=str, default='data/Tools_script/ball_counting_dataset_val.csv',
                       help='Path to the validation CSV file')
    
    # Data arguments
    parser.add_argument('--data_root', type=str,
                       default='data/ball_data_collection',
                       help='Root directory of image data')
    parser.add_argument('--train_csv', type=str, default='data/Tools_script/ball_counting_dataset_train_10.csv',
                       help='Path to the training CSV (used to build the DataLoader)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default='analysis_results/single_image',
                       help='Root directory for analysis outputs')
    
    # DataLoader arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader worker processes')
    parser.add_argument('--n_samples', type=int, default=-1,
                       help='Number of samples to use for Grad-CAM/error analysis; -1 uses all samples')
    # Visualisation arguments
    parser.add_argument('--gradcam_samples_per_class', type=int, default=2,
                       help='Number of Grad-CAM samples to visualise per class')
    parser.add_argument('--error_samples', type=int, default=20,
                       help='Number of error samples to visualise')
    
    # Device argument
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Compute device: auto (detect), cpu, or cuda (GPU)')
    
    args = parser.parse_args()

    required_paths = {
        'checkpoint': args.checkpoint,
        'data_root': args.data_root,
        'train_csv': args.train_csv,
        'val_csv': args.val_csv,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required input path(s): " + "; ".join(missing) +
            ". Pass explicit --checkpoint/--data_root/--train_csv/--val_csv values for your environment."
        )
    
    # Run analysis
    run_analysis(args)


if __name__ == '__main__':
    main()