import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class BallCountingDataset(Dataset):
    """球类计数多模态序列数据集（精简版）"""
    
    def __init__(self, csv_path, data_root, sequence_length=6, 
                 normalize_images=True, custom_image_norm_stats=None,
                 shuffle_joints=False, curriculum_mode='random', seed=42):
        """
        初始化数据集
        
        Args:
            csv_path: CSV文件路径（直接使用对应比例的CSV文件）
            data_root: 数据根目录
            sequence_length: 序列长度
            normalize_images: 是否对图像进行标准化（使用ImageNet参数）
            custom_image_norm_stats: 自定义图像标准化统计值 {"mean": [...], "std": [...]}
            shuffle_joints: 是否打乱sequence中joints的顺序（图像顺序不变）
            curriculum_mode: 课程学习模式
                - 'random': 随机顺序（默认）
                - 'easy_to_hard': 从小到大（先学小数字）
                - 'hard_to_easy': 从大到小（先学大数字）
            seed: 随机种子，用于打乱
        """
        # 读取CSV数据
        csv_data = pd.read_csv(csv_path)
        
        # 验证CSV格式
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV必须包含{col}列"
        
        print(f"加载数据: {len(csv_data)} 样本")
        
        # 根据curriculum_mode排序数据
        if curriculum_mode == 'easy_to_hard':
            csv_data = csv_data.sort_values('ball_count', ascending=True).reset_index(drop=True)
            print(f"课程学习模式: 从易到难（小数字到大数字）")
        elif curriculum_mode == 'hard_to_easy':
            csv_data = csv_data.sort_values('ball_count', ascending=False).reset_index(drop=True)
            print(f"课程学习模式: 从难到易（大数字到小数字）")
        elif curriculum_mode == 'random':
            # 随机打乱
            csv_data = csv_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"课程学习模式: 随机顺序")
        else:
            raise ValueError(f"不支持的curriculum_mode: {curriculum_mode}. 请使用 'random', 'easy_to_hard', 或 'hard_to_easy'")
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        self.shuffle_joints = shuffle_joints
        self.seed = seed
        
        if self.shuffle_joints:
            print(f"Joint打乱模式: 开启 (图像序列保持顺序，joints被打乱)")
        else:
            print(f"Joint打乱模式: 关闭 (图像和joints保持对应)")
        
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
                # 使用ImageNet标准化参数
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
            'sequence_data': sequence_data,
            'label': ball_count
        }
    
    def _load_image(self, image_path):
        """加载并处理单张RGB图像"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                return torch.zeros(3, 224, 224)
            
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_sequence_data(self, json_path):
        """加载并处理序列数据"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            original_length = len(frames)
            
            # 调整序列长度
            if len(frames) < self.sequence_length:
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[-self.sequence_length:]
            
            # 提取数据
            joints_list = []  # 只保留joint1和joint6
            labels_list = []
            images_list = []
            
            for frame in frames:
                # 提取所有关节数据
                all_joints = frame.get('joints', [0.0] * 7)
                all_joints = [float(j) if j is not None else 0.0 for j in all_joints]
                
                # 只保留joint1(索引0)和joint6(索引5)
                if len(all_joints) >= 6:
                    selected_joints = [all_joints[0], all_joints[5]]  # joint1, joint6
                else:
                    selected_joints = [0.0, 0.0]
                
                joints_list.append(selected_joints)
                
                # 提取标签
                label = frame.get('label', 0)
                labels_list.append(float(label) if label is not None else 0.0)
                
                # 处理图像路径
                image_path = frame.get('image_path', '')
                
                if image_path:
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                    else:
                        relative_image_path = image_path
                    
                    # 修复路径命名问题
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                    
                    image = self._load_image(relative_image_path)
                else:
                    image = torch.zeros(3, 224, 224)
                
                images_list.append(image)
            
            # 转换为张量
            joints = torch.tensor(joints_list, dtype=torch.float32)  # shape: [seq_len, 2]
            labels = torch.tensor(labels_list, dtype=torch.float32)
            images = torch.stack(images_list)
            
            # 如果开启shuffle_joints模式，打乱joints的顺序（保持images顺序）
            if self.shuffle_joints:
                # 生成随机排列索引
                shuffle_indices = torch.randperm(len(joints))
                joints = joints[shuffle_indices]
            
            ball_count = json_data.get('ball_count', 0)
            
            return {
                'joints': joints,  # 只包含joint1和joint6，未归一化
                'labels': labels,
                'images': images,
                'sequence_length': original_length,
                'ball_count': int(ball_count)
            }
            
        except Exception as e:
            print(f"处理JSON文件失败 {json_path}: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回默认数据
            return {
                'joints': torch.zeros(self.sequence_length, 2, dtype=torch.float32),  # 2个joint
                'labels': torch.zeros(self.sequence_length, dtype=torch.float32),
                'images': torch.zeros(self.sequence_length, 3, 224, 224),
                'sequence_length': 1,
                'ball_count': 0
            }


def get_ball_counting_data_loaders(train_csv_path, val_csv_path, data_root, 
                                   batch_size=16, sequence_length=11, 
                                   num_workers=1, normalize_images=True,
                                   custom_image_norm_stats=None,
                                   shuffle_joints=False, 
                                   curriculum_mode='random',
                                   seed=42):
    """
    创建球类计数的训练和验证数据加载器
    
    Args:
        train_csv_path: 训练集CSV路径
            - "ball_counting_dataset_train.csv" 表示100%数据
            - "ball_counting_dataset_train_10.csv" 表示10%数据
            - "ball_counting_dataset_train_50.csv" 表示50%数据
        val_csv_path: 验证集CSV路径
        data_root: 数据根目录
        batch_size: 批次大小
        sequence_length: 序列长度
        num_workers: 数据加载器进程数
        normalize_images: 是否对图像进行标准化（使用ImageNet参数）
        custom_image_norm_stats: 自定义图像标准化参数 {"mean": [...], "std": [...]}
        shuffle_joints: 是否打乱joints顺序（图像保持顺序）
        curriculum_mode: 课程学习模式
            - 'random': 随机顺序
            - 'easy_to_hard': 从小到大（先学小数字）
            - 'hard_to_easy': 从大到小（先学大数字）
        seed: 随机种子
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== 创建数据加载器 - RGB图像模式 ===")
    print("=" * 60)
    
    print("\n[训练集配置]")
    train_dataset = BallCountingDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        shuffle_joints=shuffle_joints,
        curriculum_mode=curriculum_mode,
        seed=seed
    )
    
    print("\n[验证集配置]")
    val_dataset = BallCountingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        shuffle_joints=False,  # 验证集不打乱joints
        curriculum_mode='random',  # 验证集使用随机顺序
        seed=seed
    )
    
    # 注意: 训练集不需要shuffle=True，因为数据已经按照curriculum_mode排序
    # 如果curriculum_mode='random'，数据已经被打乱；如果是easy_to_hard或hard_to_easy，需要保持顺序
    shuffle_train = (curriculum_mode == 'random')
    
    # 禁用pin_memory以避免CUDA错误（特别是在多用户共享GPU环境中）
    # 如果需要启用，可以手动设置为True
    use_pin_memory = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
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
    print(f"训练集: {len(train_dataset)} 样本, Batch shuffle: {shuffle_train}")
    print(f"验证集: {len(val_dataset)} 样本")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv_100 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    train_csv_50 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_50.csv"
    train_csv_10 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== 球类计数数据集测试（精简版 + 多种训练模式）===\n")
    
    if not os.path.exists(train_csv_100):
        print(f"错误: 训练集CSV文件不存在: {train_csv_100}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"错误: 验证集CSV文件不存在: {val_csv}")
        exit(1)
    
    try:
        # ============ 测试1: 标准模式（100%数据，随机顺序，joints不打乱）============
        print("\n" + "="*70)
        print("测试1: 标准模式 - 100%数据")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize_images=True,
            shuffle_joints=False,
            curriculum_mode='random'
        )
        
        for batch in train_loader:
            print(f"Batch shapes: Images {batch['sequence_data']['images'].shape}, "
                  f"Joints {batch['sequence_data']['joints'].shape}")
            break
        
        # ============ 测试2: Joint打乱模式 - 100%数据 ============
        print("\n" + "="*70)
        print("测试2: Joint打乱模式 - 100%数据")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            shuffle_joints=True,
            curriculum_mode='random'
        )
        
        # ============ 测试3: 从易到难 - 100%数据 ============
        print("\n" + "="*70)
        print("测试3: 课程学习 - 从易到难 - 100%数据")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='easy_to_hard'
        )
        
        # ============ 测试4: 从难到易 - 100%数据 ============
        print("\n" + "="*70)
        print("测试4: 课程学习 - 从难到易 - 100%数据")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='hard_to_easy'
        )
        
        # ============ 测试5: 使用10%数据 ============
        print("\n" + "="*70)
        print("测试5: 使用10%训练数据")
        print("="*70)
        if os.path.exists(train_csv_10):
            train_loader, val_loader = get_ball_counting_data_loaders(
                train_csv_path=train_csv_10,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                sequence_length=11
            )
        else:
            print(f"10%数据CSV不存在: {train_csv_10}")
        
        # ============ 测试6: 使用50%数据 ============
        print("\n" + "="*70)
        print("测试6: 使用50%训练数据")
        print("="*70)
        if os.path.exists(train_csv_50):
            train_loader, val_loader = get_ball_counting_data_loaders(
                train_csv_path=train_csv_50,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                sequence_length=11
            )
        else:
            print(f"50%数据CSV不存在: {train_csv_50}")
        
        # ============ 测试7: 组合模式 - Joint打乱 + 从易到难 + 10%数据 ============
        print("\n" + "="*70)
        print("测试7: 组合模式（Joint打乱 + 从易到难 + 10%数据）")
        print("="*70)
        if os.path.exists(train_csv_10):
            train_loader, val_loader = get_ball_counting_data_loaders(
                train_csv_path=train_csv_10,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                sequence_length=11,
                shuffle_joints=True,
                curriculum_mode='easy_to_hard'
            )
        
        print("\n" + "="*70)
        print("所有测试完成！")
        print("="*70)
        
        print("\n使用示例:")
        print("-" * 70)
        print("# 1. 标准训练（100%数据）")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root)")
        print()
        print("# 2. 使用10%数据")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root)")
        print()
        print("# 3. 使用50%数据")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root)")
        print()
        print("# 4. Joint打乱训练（100%数据）")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root,")
        print("    shuffle_joints=True)")
        print()
        print("# 5. 从易到难课程学习（50%数据）")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root,")
        print("    curriculum_mode='easy_to_hard')")
        print()
        print("# 6. 组合模式：Joint打乱 + 从易到难 + 10%数据")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root,")
        print("    shuffle_joints=True,")
        print("    curriculum_mode='easy_to_hard')")
        print("-" * 70)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

