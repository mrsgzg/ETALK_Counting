"""
快速测试脚本 - 验证可视化工具是否正常工作
生成模拟数据进行功能测试
"""

import torch
import numpy as np
import sys
import os

# 添加路径
CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)

from visualization_utils import (
    plot_pca, plot_tsne,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    visualize_error_samples
)

def test_visualization_tools():
    """测试所有可视化工具"""
    
    print("=" * 60)
    print("测试可视化工具")
    print("=" * 60)
    
    # 创建测试输出目录
    test_dir = 'test_visualization_output'
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 256
    n_classes = 10
    
    # 生成特征（每个类别有不同的中心）
    features = []
    labels = []
    for cls in range(n_classes):
        center = np.random.randn(n_features) * 5
        class_features = np.random.randn(n_samples // n_classes, n_features) + center
        features.append(class_features)
        labels.extend([cls + 1] * (n_samples // n_classes))  # 标签从1开始
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # 生成预测（添加一些错误）
    predictions = labels.copy()
    error_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    predictions[error_idx] = (predictions[error_idx] + np.random.randint(1, n_classes, size=len(error_idx))) % n_classes + 1
    
    # 生成模拟图像
    images = torch.randn(n_samples, 3, 224, 224)
    
    print(f"\n生成的测试数据:")
    print(f"  样本数: {n_samples}")
    print(f"  特征维度: {n_features}")
    print(f"  类别数: {n_classes}")
    print(f"  准确率: {(predictions == labels).mean():.2%}")
    
    # 测试1: PCA 2D
    print("\n[1/7] 测试 PCA 2D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_2d.png'),
                n_components=2,
                title="Test PCA 2D")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试2: PCA 3D
    print("\n[2/7] 测试 PCA 3D...")
    try:
        plot_pca(features, labels,
                save_path=os.path.join(test_dir, 'test_pca_3d.png'),
                n_components=3,
                title="Test PCA 3D")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试3: t-SNE 2D
    print("\n[3/7] 测试 t-SNE 2D...")
    try:
        plot_tsne(features, labels,
                 save_path=os.path.join(test_dir, 'test_tsne_2d.png'),
                 n_components=2,
                 perplexity=30,
                 title="Test t-SNE 2D")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试4: 混淆矩阵
    print("\n[4/7] 测试混淆矩阵...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=False)
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试5: 归一化混淆矩阵
    print("\n[5/7] 测试归一化混淆矩阵...")
    try:
        plot_confusion_matrix(labels, predictions,
                            save_path=os.path.join(test_dir, 'test_confusion_matrix_norm.png'),
                            class_names=[str(i) for i in range(1, 11)],
                            normalize=True)
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试6: 每类准确率
    print("\n[6/7] 测试每类准确率...")
    try:
        plot_per_class_accuracy(labels, predictions,
                               save_path=os.path.join(test_dir, 'test_per_class_acc.png'),
                               class_names=[str(i) for i in range(1, 11)])
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试7: 错误样本可视化
    print("\n[7/7] 测试错误样本可视化...")
    try:
        visualize_error_samples(
            images=images,
            labels=torch.from_numpy(labels),
            predictions=torch.from_numpy(predictions),
            save_path=os.path.join(test_dir, 'test_error_samples.png'),
            n_samples=20
        )
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)
    print(f"\n所有测试图片已保存到: {test_dir}/")
    print("\n生成的文件:")
    for f in os.listdir(test_dir):
        print(f"  - {f}")
    print()

if __name__ == '__main__':
    test_visualization_tools()
