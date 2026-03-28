import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SingleImageDataset(Dataset):
    """单图像分类数据集 - 从序列数据中提取所有帧（包括填充帧）"""
    
    def __init__(self, csv_path, data_root, 
                 sequence_length=11,
                 normalize_images=True, 
                 custom_image_norm_stats=None):
        """
        初始化单图像数据集
        
        Args:
            csv_path: CSV文件路径
            data_root: 数据根目录
            sequence_length: 序列长度（用于填充）
            normalize_images: 是否对图像进行标准化（使用ImageNet参数）
            custom_image_norm_stats: 自定义图像标准化统计值 {"mean": [...], "std": [...]}
        """
        # 读取CSV数据
        csv_data = pd.read_csv(csv_path)
        
        # 验证CSV格式
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV必须包含{col}列"
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # 设置图像变换
        self._setup_image_transforms()
        
        # 构建单图像样本列表
        self.samples = self._build_sample_list()
        
        print(f"单图像数据集构建完成:")
        print(f"  原始序列数: {len(self.csv_data)}")
        print(f"  提取的单图像样本数: {len(self.samples)}")
        print(f"  图像模式: RGB")
        print(f"  标签: ball_count (1-10)")
    
    def _setup_image_transforms(self):
        """设置图像变换流水线（RGB模式）"""
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
        
        # 图像标准化
        if self.normalize_images:
            if self.custom_image_norm_stats:
                mean = self.custom_image_norm_stats["mean"]
                std = self.custom_image_norm_stats["std"]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            else:
                # ImageNet标准化参数
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
        print(f"图像处理: RGB模式, 标准化: {self.normalize_images}")
    
    def _build_sample_list(self):
        """构建单图像样本列表 - 提取所有帧（包括填充后的帧）"""
        samples = []
        
        for idx in range(len(self.csv_data)):
            sample_row = self.csv_data.iloc[idx]
            sample_id = sample_row['sample_id']
            ball_count = sample_row['ball_count']
            json_path = sample_row['json_path']
            
            try:
                # 加载JSON数据
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                frames = json_data['frames']
                original_length = len(frames)
                
                # 调整序列长度（与embodiment模型保持一致）
                if len(frames) < self.sequence_length:
                    # 填充：重复最后一帧
                    last_frame = frames[-1]
                    frames = frames + [last_frame] * (self.sequence_length - len(frames))
                elif len(frames) > self.sequence_length:
                    frames = frames[:self.sequence_length]
                
                # 提取所有帧（包括填充的重复帧）
                for frame_idx, frame in enumerate(frames):
                    # 获取图像路径
                    image_path = frame.get('image_path', '')
                    if image_path:
                        # 处理图像路径
                        path_parts = image_path.split('/')
                        if 'ball_data_collection' in path_parts:
                            ball_data_idx = path_parts.index('ball_data_collection')
                            relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                        else:
                            relative_image_path = image_path
                        
                        # 修复路径命名不一致问题
                        if '1_ball' in relative_image_path:
                            relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                        
                        # 创建样本
                        sample = {
                            'image_path': relative_image_path,
                            'label': int(ball_count),
                            'sample_id': sample_id,
                            'frame_idx': frame_idx
                        }
                        samples.append(sample)
                
            except Exception as e:
                print(f"处理JSON文件失败 {json_path}: {e}")
                continue
        
        return samples
    
    def _load_image(self, image_path):
        """加载并处理单张RGB图像"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"图像不存在: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            # 加载图像
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'label': sample['label'],
            'sample_id': sample['sample_id'],
            'frame_idx': sample['frame_idx']
        }
    
    def get_class_distribution(self):
        """获取类别分布"""
        labels = [sample['label'] for sample in self.samples]
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


def get_single_image_data_loaders(train_csv_path, val_csv_path, data_root, 
                                  batch_size=32, 
                                  sequence_length=11,
                                  num_workers=4, 
                                  normalize_images=True,
                                  custom_image_norm_stats=None):
    """
    创建单图像分类的训练和验证数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
        val_csv_path: 验证集CSV路径
        data_root: 数据根目录
        batch_size: 批次大小
        sequence_length: 序列长度（用于填充）
        num_workers: 数据加载器进程数
        normalize_images: 是否对图像进行标准化（使用ImageNet参数）
        custom_image_norm_stats: 自定义图像标准化参数 {"mean": [...], "std": [...]}
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== 创建单图像数据加载器 - RGB图像模式 ===")
    print("=" * 60)
    
    print("=" * 60)
    print("=== 创建单图像数据加载器 - RGB图像模式 ===")
    print("=" * 60)
    
    print("\n[训练集配置]")
    # 创建训练集
    train_dataset = SingleImageDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[验证集配置]")
    # 创建验证集
    val_dataset = SingleImageDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_dist = train_dataset.get_class_distribution()
    for label in sorted(train_dist.keys()):
        print(f"  球数 {label}: {train_dist[label]} 样本")
    
    print("\n验证集类别分布:")
    val_dist = val_dataset.get_class_distribution()
    for label in sorted(val_dist.keys()):
        print(f"  球数 {label}: {val_dist[label]} 样本")
    
    # 禁用pin_memory以避免CUDA错误
    use_pin_memory = True
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print("\n" + "=" * 60)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 数据集路径配置
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== 单图像分类数据集测试 ===")
    
    # 检查文件是否存在
    if not os.path.exists(train_csv):
        print(f"错误: 训练集CSV文件不存在: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"错误: 验证集CSV文件不存在: {val_csv}")
        exit(1)
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        exit(1)
    
    try:
        # 测试数据加载器
        train_loader, val_loader = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize_images=True
        )
        
        print(f"训练集样本数: {len(train_loader.dataset)}")
        print(f"验证集样本数: {len(val_loader.dataset)}")
        
        # 测试batch数据
        for batch in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Labels: {batch['label'].shape}")
            print(f"  图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"  标签范围: [{batch['label'].min()}, {batch['label'].max()}]")
            print(f"  样本标签示例: {batch['label'][:5].tolist()}")
            break
        
        print("\n=== 使用示例 ===")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=32, sequence_length=11)")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()