"""
info - info
info
"""

import torch
import numpy as np
import sys
import os

# info
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)

from visualization_utils import (
    plot_pca, plot_tsne,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    visualize_error_samples
)

def test_visualization_tools():
    """info"""
    
    print("=" * 60)
    print("info")
    print("=" * 60)
    
    # info
    test_dir = 'test_visualization_output'
    os.makedirs(test_dir, exist_ok=True)
    
    # info
    np.random.seed(42)
    n_samples = 200
    n_features = 256
    n_classes = 10
    
    # infoïinfoïinfo
    features = []
    labels = []
    for cls in range(n_classes):
        center = np.random.randn(n_features) * 5
        class_features = np.random.randn(n_samples // n_classes, n_features) + center
        features.append(class_features)
        labels.extend([cls + 1] * (n_samples // n_classes))  # Step 1
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # infoïinfoïinfo
    predictions = labels.copy()
    error_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    predictions[error_idx] = (predictions[error_idx] + np.random.randint(1, n_classes, size=len(error_idx))) % n_classes + 1
    
    # info
    images = torch.randn(n_samples, 3, 224, 224)
    
    print(f"\n:")
    print(f"  info: {n_samples}")
    print(f"  info: {n_features}")
    print(f"  info: {n_classes}")
    print(f"  info: {(predictions == labels).mean():.2%}")
    
    # Step 1: PCA 2D
    print("\n[1/7] info PCA 2D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_2d.png'),
                n_components=2,
                title="Test PCA 2D")
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 2: PCA 3D
    print("\n[2/7] info PCA 3D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_3d.png'),
                n_components=3,
                title="Test PCA 3D")
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 3: t-SNE 2D
    print("\n[3/7] info t-SNE 2D...")
    try:
        plot_tsne(features, labels,
                 save_path=os.path.join(test_dir, 'test_tsne_2d.png'),
                 n_components=2,
                 perplexity=30,
                 title="Test t-SNE 2D")
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 4: info
    print("\n[4/7] info...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=False)
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 5: info
    print("\n[5/7] info...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix_norm.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=True)
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 6: info
    print("\n[6/7] info...")
    try:
        plot_per_class_accuracy(labels, predictions,
                               save_path=os.path.join(test_dir, 'test_per_class_acc.png'),
                               class_names=[str(i) for i in range(1, 11)])
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # Step 7: info
    print("\n[7/7] info...")
    try:
        visualize_error_samples(
            images=images,
            labels=torch.from_numpy(labels),
            predictions=torch.from_numpy(predictions),
            save_path=os.path.join(test_dir, 'test_error_samples.png'),
            n_samples=20
        )
        print("  info info")
    except Exception as e:
        print(f"  info info: {e}")
    
    # info
    print("\n" + "=" * 60)
    print("info infoïinfo")
    print("=" * 60)
    print(f"\n: {test_dir}/")
    print("\n:")
    for f in os.listdir(test_dir):
        print(f"  - {f}")
    print()

if __name__ == '__main__':
    test_visualization_tools()
