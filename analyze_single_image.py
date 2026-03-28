"""
info - info
info: PCA/t-SNEGrad-CAM
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

# info
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
    checkpoint
    
    Args:
        checkpoint_path: checkpoint
        device: info
    
    Returns:
        model: info
        checkpoint: checkpoint
    """
    print(f"\n{'='*60}")
    print(f"info: {checkpoint_path}")
    print(f"{'='*60}")
    
    # checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # info
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config.get('num_classes', 11)
        use_pretrain = config.get('use_pretrain', True)
        print(f"  info:")
        print(f"    info: {num_classes}")
        print(f"    info: {use_pretrain}")
        if 'epoch' in checkpoint:
            print(f"    info: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"    info: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"    info: {checkpoint['val_acc']:.2%}")
    else:
        # info
        num_classes = 11
        use_pretrain = True
        print(f"  info: num_classes={num_classes}, use_pretrain={use_pretrain}")
    
    # info
    model = create_single_image_model(
        num_classes=num_classes,
        use_pretrain=False,  # infoïcheckpoint
        input_channels=3
    )
    
    # info - info
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # checkpointstate_dict
        # info
        filtered_state = {k: v for k, v in checkpoint.items() 
                         if k in model.state_dict()}
        if filtered_state:
            model.load_state_dict(filtered_state)
        else:
            raise ValueError(f"Unable to load model state. Checkpoint keys: {list(checkpoint.keys())}")
    
    model = model.to(device)
    model.eval()
    
    print(f"info infoïinfo")
    print(f"{'='*60}\n")
    
    return model, checkpoint


def extract_features_and_predictions(model, dataloader, device):
    """
    info
    
    Args:
        model: info
        dataloader: info
        device: info
    
    Returns:
        features: [N, D] info
        labels: [N] info
        predictions: [N] info
        images: [N, C, H, W] info
        logits: [N, num_classes] logits
    """
    print(f"\n{'='*60}")
    print("info...")
    print(f"{'='*60}")
    
    all_features = []
    all_labels = []
    all_predictions = []
    all_images = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="info"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # info (AlexNet encoder)
            features = model.visual_encoder(images)  # [B, 256]
            
            # info
            logits = model.classifier(features)  # [B, num_classes]
            predictions = logits.argmax(dim=1)
            
            # info
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_images.append(images.cpu())
            all_logits.append(logits.cpu())
    
    # info
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    images = torch.cat(all_images, dim=0)
    logits = torch.cat(all_logits, dim=0).numpy()
    
    # info
    accuracy = (predictions == labels).mean()
    
    print(f"\n:")
    print(f"  info: {len(features)}")
    print(f"  info: {features.shape[1]}")
    print(f"  info: {accuracy:.2%}")
    print(f"{'='*60}\n")
    
    return features, labels, predictions, images, logits


def run_analysis(args):
    """
    info
    
    Args:
        args: info
    """
    # info
    if args.device.lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device.lower())
    print(f"info: {device}")
    
    # info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    feature_dir = os.path.join(output_dir, 'features')
    attention_dir = os.path.join(output_dir, 'attention_maps')
    prediction_dir = os.path.join(output_dir, 'predictions')
    
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    
    print(f"\n: {output_dir}\n")
    
    # 1. info
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # 2. info
    print(f"{'='*60}")
    print("info...")
    print(f"{'='*60}")
    
    train_loader, val_loader = get_single_image_data_loaders(
        train_csv_path=args.train_csv if args.train_csv else 'scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv',
        val_csv_path=args.val_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=11,
        normalize_images=True
    )
    
    # CPUïpin_memoryCUDA
    use_pin_memory = device.type == 'cuda'
    
    # val_loaderpin_memoryïCPUïinfo
    if not use_pin_memory:
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False
        )
    
    # info
    dataloader_full = val_loader
    print(f"info infoïinfo: {len(dataloader_full.dataset)}\n")
    
    # infoïPCA/t-SNEïinfo
    print("infoïinfoïPCA/t-SNEïinfo")
    features_full, labels_full, predictions_full, images_full, logits_full = extract_features_and_predictions(
        model, dataloader_full, device
    )
    
    # infoïGrad-CAMïinfo
    if args.n_samples > 0:
        # info
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
        print(f"infoïinfoïGrad-CAM/infoïinfo")
        features_subset, labels_subset, predictions_subset, images_subset, logits_subset = extract_features_and_predictions(
            model, dataloader_subset, device
        )
        print(f"info info: {len(dataloader_subset.dataset)}\n")
    else:
        # info
        features_subset = features_full
        labels_subset = labels_full
        predictions_subset = predictions_full
        images_subset = images_full
        logits_subset = logits_full
    
    # 4. infoïinfoïinfo
    print(f"{'='*60}")
    print("infoïinfoïinfo...")
    print(f"{'='*60}\n")
    
    # PCA 2D
    plot_pca(
        features_full, labels_full,
        save_path=os.path.join(feature_dir, 'pca_2d.png'),
        n_components=2,
        title="PCA 2D Visualization (All Validation Data)"
    )

    # PCACSVïStep 2DStep 3Dïinfo
    try:
        from sklearn.decomposition import PCA
        # 2D PCA
        pca_2d = PCA(n_components=2)
        features_pca_2d = pca_2d.fit_transform(features_full)
        # info [pc1, pc2, label, prediction]
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

        # 3D PCAïinfoïinfo
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
        print(f"info PCA CSV: {e}")
    
    # t-SNE 2D
    if len(features_full) <= 5000:  # t-SNE
        plot_tsne(
            features_full, labels_full,
            save_path=os.path.join(feature_dir, 'tsne_2d.png'),
            n_components=2,
            perplexity=min(30, len(features_full) // 5),
            title="t-SNE 2D Visualization (All Validation Data)"
        )
    else:
        print(f"info info({len(features_full)})ït-SNEïinfo<5000ïinfo")
    
    print()
    
    # 5. Grad-CAMïStep 2ïinfo
    print(f"{'='*60}")
    print("Grad-CAMïStep 2ïinfo...")
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
    
    # 6. info
    print(f"{'='*60}")
    print("info...")
    print(f"{'='*60}\n")
    
    # info
    class_names = [str(i) for i in range(1, 11)]  # 1-10
    
    # info
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix.png'),
        class_names=class_names,
        normalize=False
    )
    
    # info
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix_normalized.png'),
        class_names=class_names,
        normalize=True
    )
    
    # info
    plot_per_class_accuracy(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'per_class_accuracy.png'),
        class_names=class_names
    )
    
    # info
    visualize_error_samples(
        images=images_subset,
        labels=torch.from_numpy(labels_subset),
        predictions=torch.from_numpy(predictions_subset),
        save_path=os.path.join(prediction_dir, 'error_samples.png'),
        n_samples=args.error_samples
    )
    
    # 7. Softmax
    print(f"{'='*60}")
    print("Softmax...")
    print(f"{'='*60}\n")
    
    visualize_softmax_outputs(
        logits=logits_subset,
        labels=labels_subset,
        predictions=predictions_subset,
        save_path=os.path.join(prediction_dir, 'softmax_outputs.png'),
        class_names=class_names
    )
    
    print()
    
    # 8. info
    print(f"{'='*60}")
    print("info...")
    print(f"{'='*60}\n")
    
    # infoïinfoïinfo
    per_class_acc = {}
    unique_labels = np.unique(labels_full)
    for label in unique_labels:
        mask = (labels_full == label)
        acc = (predictions_full[mask] == label).mean()
        per_class_acc[int(label)] = acc
    
    # PCAïinfoïinfo
    from sklearn.decomposition import PCA
    # Step 3D PCAïinfoïinfo
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
    
    # info
    print(f"\n{'='*60}")
    print("info infoïinfo")
    print(f"{'='*60}")
    print(f"\n: {output_dir}")
    print(f"\n:")
    print(f"  {output_dir}/")
    print(f"    info features/           # PCAt-SNE")
    print(f"    info attention_maps/     # Grad-CAM")
    print(f"    info predictions/        # info")
    print(f"    info analysis_summary.txt  # info")
    print()


def main():
    parser = argparse.ArgumentParser(description='info')
    
    # info
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='checkpoint')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='CSV')
    
    # info
    parser.add_argument('--data_root', type=str,
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='info')
    parser.add_argument('--train_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv',
                       help='CSVïinfoïinfoïinfo')
    
    # info
    parser.add_argument('--output_dir', type=str, 
                       default='scratch/Cognitive_Embodied_Counting/Visualization/new_singelimage',
                       help='info')
    
    # info
    parser.add_argument('--batch_size', type=int, default=32,
                       help='info')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='info')
    parser.add_argument('--n_samples', type=int, default=-1,
                       help='infoïinfo-1')
    # info
    parser.add_argument('--gradcam_samples_per_class', type=int, default=2,
                       help='classGrad-CAM')
    parser.add_argument('--error_samples', type=int, default=20,
                       help='info')
    
    # info
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['auto', 'cpu', 'cuda'],
                       help='info: auto (info), cpu (CPU), cuda (GPU)')
    
    args = parser.parse_args()
    
    # info
    run_analysis(args)


if __name__ == '__main__':
    main()
