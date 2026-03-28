"""
单图像模型分析脚本 - 可视化与可解释性分析
功能: PCA/t-SNE降维、Grad-CAM热图、混淆矩阵、错误分析
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 添加路径
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
    加载模型checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    Returns:
        model: 加载的模型
        checkpoint: checkpoint字典
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {checkpoint_path}")
    print(f"{'='*60}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config.get('num_classes', 11)
        use_pretrain = config.get('use_pretrain', True)
        print(f"  配置信息:")
        print(f"    类别数: {num_classes}")
        print(f"    预训练: {use_pretrain}")
        if 'epoch' in checkpoint:
            print(f"    训练轮次: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"    验证损失: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"    验证准确率: {checkpoint['val_acc']:.2%}")
    else:
        # 默认配置
        num_classes = 11
        use_pretrain = True
        print(f"  使用默认配置: num_classes={num_classes}, use_pretrain={use_pretrain}")
    
    # 创建模型
    model = create_single_image_model(
        num_classes=num_classes,
        use_pretrain=False,  # 不再需要预训练权重，直接加载checkpoint
        input_channels=3
    )
    
    # 加载模型参数 - 尝试不同的键名
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 假设整个checkpoint就是state_dict
        # 过滤掉非模型参数的键
        filtered_state = {k: v for k, v in checkpoint.items() 
                         if k in model.state_dict()}
        if filtered_state:
            model.load_state_dict(filtered_state)
        else:
            raise ValueError(f"Unable to load model state. Checkpoint keys: {list(checkpoint.keys())}")
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功！")
    print(f"{'='*60}\n")
    
    return model, checkpoint


def extract_features_and_predictions(model, dataloader, device):
    """
    提取特征、预测和标签
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        features: [N, D] 特征矩阵
        labels: [N] 真实标签
        predictions: [N] 预测标签
        images: [N, C, H, W] 图像张量
        logits: [N, num_classes] 预测logits
    """
    print(f"\n{'='*60}")
    print("提取特征和预测...")
    print(f"{'='*60}")
    
    all_features = []
    all_labels = []
    all_predictions = []
    all_images = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="处理批次"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 提取特征 (AlexNet encoder输出)
            features = model.visual_encoder(images)  # [B, 256]
            
            # 获取预测
            logits = model.classifier(features)  # [B, num_classes]
            predictions = logits.argmax(dim=1)
            
            # 保存
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_images.append(images.cpu())
            all_logits.append(logits.cpu())
    
    # 合并
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    images = torch.cat(all_images, dim=0)
    logits = torch.cat(all_logits, dim=0).numpy()
    
    # 计算准确率
    accuracy = (predictions == labels).mean()
    
    print(f"\n特征提取完成:")
    print(f"  样本数: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  准确率: {accuracy:.2%}")
    print(f"{'='*60}\n")
    
    return features, labels, predictions, images, logits


def run_analysis(args):
    """
    运行完整的可视化分析
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    if args.device.lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device.lower())
    print(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    feature_dir = os.path.join(output_dir, 'features')
    attention_dir = os.path.join(output_dir, 'attention_maps')
    prediction_dir = os.path.join(output_dir, 'predictions')
    
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}\n")
    
    # 1. 加载模型
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # 2. 加载数据
    print(f"{'='*60}")
    print("加载数据集...")
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
    
    # 如果使用CPU，禁用pin_memory以避免CUDA错误
    use_pin_memory = device.type == 'cuda'
    
    # 重新创建val_loader以禁用pin_memory（如果使用CPU）
    if not use_pin_memory:
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False
        )
    
    # 使用验证集
    dataloader_full = val_loader
    print(f"✓ 使用验证集，总样本数: {len(dataloader_full.dataset)}\n")
    
    # 提取全量数据特征（用于PCA/t-SNE）
    print("第一阶段：提取全量验证数据特征（用于PCA/t-SNE）")
    features_full, labels_full, predictions_full, images_full, logits_full = extract_features_and_predictions(
        model, dataloader_full, device
    )
    
    # 限制样本数量（用于Grad-CAM和其他详细分析）
    if args.n_samples > 0:
        # 创建子集
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
        print(f"第二阶段：提取限制数据特征（用于Grad-CAM/详细分析）")
        features_subset, labels_subset, predictions_subset, images_subset, logits_subset = extract_features_and_predictions(
            model, dataloader_subset, device
        )
        print(f"✓ 限制样本数: {len(dataloader_subset.dataset)}\n")
    else:
        # 使用全量数据
        features_subset = features_full
        labels_subset = labels_full
        predictions_subset = predictions_full
        images_subset = images_full
        logits_subset = logits_full
    
    # 4. 特征空间可视化（使用全量验证数据）
    print(f"{'='*60}")
    print("特征空间可视化（全量验证数据）...")
    print(f"{'='*60}\n")
    
    # PCA 2D
    plot_pca(
        features_full, labels_full,
        save_path=os.path.join(feature_dir, 'pca_2d.png'),
        n_components=2,
        title="PCA 2D Visualization (All Validation Data)"
    )

    # 将PCA降维后的数据保存为CSV（2D与3D）
    try:
        from sklearn.decomposition import PCA
        # 2D PCA
        pca_2d = PCA(n_components=2)
        features_pca_2d = pca_2d.fit_transform(features_full)
        # 合并为 [pc1, pc2, label, prediction]
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

        # 3D PCA（同时用于后续方差统计）
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
        print(f"⚠ 保存PCA CSV失败: {e}")
    
    # t-SNE 2D
    if len(features_full) <= 5000:  # t-SNE对大数据集较慢
        plot_tsne(
            features_full, labels_full,
            save_path=os.path.join(feature_dir, 'tsne_2d.png'),
            n_components=2,
            perplexity=min(30, len(features_full) // 5),
            title="t-SNE 2D Visualization (All Validation Data)"
        )
    else:
        print(f"⚠ 样本数过多({len(features_full)})，跳过t-SNE分析（建议<5000）")
    
    print()
    
    # 5. Grad-CAM可视化（每类2个样本）
    print(f"{'='*60}")
    print("Grad-CAM可视化（每类2个样本）...")
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
    
    # 6. 预测分析
    print(f"{'='*60}")
    print("预测分析...")
    print(f"{'='*60}\n")
    
    # 类别名称
    class_names = [str(i) for i in range(1, 11)]  # 1-10个球
    
    # 混淆矩阵
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix.png'),
        class_names=class_names,
        normalize=False
    )
    
    # 归一化混淆矩阵
    plot_confusion_matrix(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'confusion_matrix_normalized.png'),
        class_names=class_names,
        normalize=True
    )
    
    # 每类准确率
    plot_per_class_accuracy(
        labels_subset, predictions_subset,
        save_path=os.path.join(prediction_dir, 'per_class_accuracy.png'),
        class_names=class_names
    )
    
    # 错误样本可视化
    visualize_error_samples(
        images=images_subset,
        labels=torch.from_numpy(labels_subset),
        predictions=torch.from_numpy(predictions_subset),
        save_path=os.path.join(prediction_dir, 'error_samples.png'),
        n_samples=args.error_samples
    )
    
    # 7. Softmax输出可视化
    print(f"{'='*60}")
    print("Softmax输出可视化...")
    print(f"{'='*60}\n")
    
    visualize_softmax_outputs(
        logits=logits_subset,
        labels=labels_subset,
        predictions=predictions_subset,
        save_path=os.path.join(prediction_dir, 'softmax_outputs.png'),
        class_names=class_names
    )
    
    print()
    
    # 8. 保存分析摘要
    print(f"{'='*60}")
    print("生成分析摘要...")
    print(f"{'='*60}\n")
    
    # 计算每类准确率（使用全量数据）
    per_class_acc = {}
    unique_labels = np.unique(labels_full)
    for label in unique_labels:
        mask = (labels_full == label)
        acc = (predictions_full[mask] == label).mean()
        per_class_acc[int(label)] = acc
    
    # 计算PCA方差（使用全量数据）
    from sklearn.decomposition import PCA
    # 若之前已计算3D PCA，则复用；否则计算一次
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
    
    # 完成
    print(f"\n{'='*60}")
    print("✓ 分析完成！")
    print(f"{'='*60}")
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── features/           # PCA、t-SNE可视化")
    print(f"    ├── attention_maps/     # Grad-CAM热图")
    print(f"    ├── predictions/        # 混淆矩阵、错误分析")
    print(f"    └── analysis_summary.txt  # 分析摘要")
    print()


def main():
    parser = argparse.ArgumentParser(description='单图像模型可视化分析')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint文件路径')
    parser.add_argument('--val_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv',
                       help='验证集CSV文件路径')
    
    # 数据路径
    parser.add_argument('--data_root', type=str,
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection',
                       help='数据根目录')
    parser.add_argument('--train_csv', type=str, default='scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv',
                       help='训练集CSV文件路径（可选，默认只使用验证集）')
    
    # 输出设置
    parser.add_argument('--output_dir', type=str, 
                       default='scratch/Cognitive_Embodied_Counting/Visualization/new_singelimage',
                       help='输出目录')
    
    # 数据加载
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--n_samples', type=int, default=-1,
                       help='分析的样本数量，-1表示使用全部数据')
    # 可视化设置
    parser.add_argument('--gradcam_samples_per_class', type=int, default=2,
                       help='每个class的Grad-CAM样本数量')
    parser.add_argument('--error_samples', type=int, default=20,
                       help='错误样本可视化数量')
    
    # 设备设置
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['auto', 'cpu', 'cuda'],
                       help='推理设备: auto (自动), cpu (强制CPU), cuda (GPU)')
    
    args = parser.parse_args()
    
    # 运行分析
    run_analysis(args)


if __name__ == '__main__':
    main()
