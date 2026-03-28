import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SequencePoolingDataset(Dataset):
    """序列图像数据集 - 用于序列池化分类（无joints）"""
    
    def __init__(self, csv_path, data_root, sequence_length=11, 
                 normalize_images=True, custom_image_norm_stats=None):
        """
        初始化序列池化数据集
        
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
        
        print(f"加载数据: {len(csv_data)} 样本")
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # 设置图像变换
        self._setup_image_transforms()
    
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
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample_row = self.csv_data.iloc[idx]
        sample_id = sample_row['sample_id']
        ball_count = sample_row['ball_count']
        json_path = sample_row['json_path']
        
        sequence_data = self._load_sequence_data(json_path)
        
        return {
            'sample_id': sample_id,
            'images': sequence_data,  # [seq_len, C, H, W]
            'label': ball_count
        }
    
    def _load_image(self, image_path):
        """加载并处理单张RGB图像"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"图像不存在: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_sequence_data(self, json_path):
        """加载并处理序列数据（只加载图像）"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            
            # 调整序列长度（与embodiment模型保持一致）
            if len(frames) < self.sequence_length:
                # 填充：重复最后一帧
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[:self.sequence_length]
            
            # 提取图像序列
            images_list = []
            for frame in frames:
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
                    
                    image = self._load_image(relative_image_path)
                    images_list.append(image)
                else:
                    # 如果没有图像路径，填充零图像
                    images_list.append(torch.zeros(3, 224, 224))
            
            # 堆叠成张量 [seq_len, C, H, W]
            images_tensor = torch.stack(images_list, dim=0)
            
            return images_tensor
            
        except Exception as e:
            print(f"加载序列数据失败 {json_path}: {e}")
            # 返回全零序列
            return torch.zeros(self.sequence_length, 3, 224, 224)


def get_sequence_pooling_data_loaders(train_csv_path, val_csv_path, data_root, 
                                      batch_size=16, sequence_length=11, 
                                      num_workers=1, normalize_images=True,
                                      custom_image_norm_stats=None):
    """
    创建序列池化分类的训练和验证数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
        val_csv_path: 验证集CSV路径
        data_root: 数据根目录
        batch_size: 批次大小
        sequence_length: 序列长度
        num_workers: 数据加载器进程数
        normalize_images: 是否对图像进行标准化（使用ImageNet参数）
        custom_image_norm_stats: 自定义图像标准化参数 {"mean": [...], "std": [...]}
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== 创建序列池化数据加载器 - RGB图像模式 ===")
    print("=" * 60)
    
    print("\n[训练集配置]")
    train_dataset = SequencePoolingDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[验证集配置]")
    val_dataset = SequencePoolingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # 禁用pin_memory以避免CUDA错误
    use_pin_memory = True
    
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
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== 序列池化数据集测试 ===\n")
    
    if not os.path.exists(train_csv):
        print(f"错误: 训练集CSV文件不存在: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"错误: 验证集CSV文件不存在: {val_csv}")
        exit(1)
    
    try:
        train_loader, val_loader = get_sequence_pooling_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize_images=True
        )
        
        # 测试batch数据
        for batch in train_loader:
            print(f"Batch shapes:")
            print(f"  Images: {batch['images'].shape}")  # [B, 11, C, H, W]
            print(f"  Labels: {batch['label'].shape}")
            print(f"  图像值范围: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
            print(f"  标签范围: [{batch['label'].min()}, {batch['label'].max()}]")
            break
        
        print("\n=== 使用示例 ===")
        print("train_loader, val_loader = get_sequence_pooling_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=16, sequence_length=11)")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
