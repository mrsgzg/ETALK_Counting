"""
Visualization tools test script.
Generates synthetic data and runs all visualization functions to verify they work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add script directory to path
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)

from visualization_utils import (
    plot_pca, plot_tsne,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    visualize_error_samples
)

def test_visualization_tools():
    """Run all visualization function tests with synthetic data."""
    
    print("=" * 60)
    print("Visualization Tools Test")
    print("=" * 60)
    
    # Create output directory for test results
    test_dir = 'test_visualization_output'
    os.makedirs(test_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 200
    n_features = 256
    n_classes = 10
    
    # Generate synthetic features with cluster structure (one cluster per class)
    features = []
    labels = []
    for cls in range(n_classes):
        center = np.random.randn(n_features) * 5
        class_features = np.random.randn(n_samples // n_classes, n_features) + center
        features.append(class_features)
        labels.extend([cls + 1] * (n_samples // n_classes))  # labels 1-10
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # Generate predictions with ~10% random errors
    predictions = labels.copy()
    error_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    predictions[error_idx] = (predictions[error_idx] + np.random.randint(1, n_classes, size=len(error_idx))) % n_classes + 1
    
    # Synthetic images (random tensors for testing error sample visualisation)
    images = torch.randn(n_samples, 3, 224, 224)
    
    print(f"\nTest data summary:")
    print(f"  Samples: {n_samples}")
    print(f"  Feature dim: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"  Accuracy: {(predictions == labels).mean():.2%}")
    
    # Test 1: PCA 2D
    print("\n[1/7] Testing PCA 2D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_2d.png'),
                n_components=2,
                title="Test PCA 2D")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: PCA 3D
    print("\n[2/7] Testing PCA 3D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_3d.png'),
                n_components=3,
                title="Test PCA 3D")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 3: t-SNE 2D
    print("\n[3/7] Testing t-SNE 2D...")
    try:
        plot_tsne(features, labels,
                 save_path=os.path.join(test_dir, 'test_tsne_2d.png'),
                 n_components=2,
                 perplexity=30,
                 title="Test t-SNE 2D")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 4: Confusion matrix (raw counts)
    print("\n[4/7] Testing confusion matrix (raw)...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=False)
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 5: Confusion matrix (normalised)
    print("\n[5/7] Testing confusion matrix (normalised)...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix_norm.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=True)
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 6: Per-class accuracy bar chart
    print("\n[6/7] Testing per-class accuracy plot...")
    try:
        plot_per_class_accuracy(labels, predictions,
                               save_path=os.path.join(test_dir, 'test_per_class_acc.png'),
                               class_names=[str(i) for i in range(1, 11)])
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 7: Error sample visualisation
    print("\n[7/7] Testing error sample visualisation...")
    try:
        visualize_error_samples(
            images=images,
            labels=torch.from_numpy(labels),
            predictions=torch.from_numpy(predictions),
            save_path=os.path.join(test_dir, 'test_error_samples.png'),
            n_samples=20
        )
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
    print(f"\nOutputs saved to: {test_dir}/")
    print("\nGenerated files:")
    for f in os.listdir(test_dir):
        print(f"  - {f}")
    print()

if __name__ == '__main__':
    test_visualization_tools()