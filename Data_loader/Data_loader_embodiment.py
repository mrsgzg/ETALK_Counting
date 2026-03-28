import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class BallCountingDataset(Dataset):
    """infoïinfoïinfo"""
    
    def __init__(self, csv_path, data_root, sequence_length=6, 
                 normalize_images=True, custom_image_norm_stats=None,
                 shuffle_joints=False, curriculum_mode='random', seed=42):
        """
        info
        
        Args:
            csv_path: CSVïCSVïinfo
            data_root: info
            sequence_length: info
            normalize_images: infoïImageNet pretrained weightsïinfo
            custom_image_norm_stats: info {"mean": [...], "std": [...]}
            shuffle_joints: sequencejointsïinfoïinfo
            curriculum_mode: info
                - 'random': infoïinfoïinfo
                - 'easy_to_hard': infoïinfoïinfo
                - 'hard_to_easy': infoïinfoïinfo
            seed: infoïinfo
        """
        # CSV
        csv_data = pd.read_csv(csv_path)
        
        # CSV
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV{col}info"
        
        print(f"info: {len(csv_data)} info")
        
        # curriculum_mode
        if curriculum_mode == 'easy_to_hard':
            csv_data = csv_data.sort_values('ball_count', ascending=True).reset_index(drop=True)
            print(f"info: infoïinfoïinfo")
        elif curriculum_mode == 'hard_to_easy':
            csv_data = csv_data.sort_values('ball_count', ascending=False).reset_index(drop=True)
            print(f"info: infoïinfoïinfo")
        elif curriculum_mode == 'random':
            # info
            csv_data = csv_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"info: info")
        else:
            raise ValueError(f"curriculum_mode: {curriculum_mode}. info 'random', 'easy_to_hard', info 'hard_to_easy'")
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        self.shuffle_joints = shuffle_joints
        self.seed = seed
        
        if self.shuffle_joints:
            print(f"Joint: info (infoïjoints)")
        else:
            print(f"Joint: info (joints)")
        
        # info
        self._setup_image_transforms()
    
    def _setup_image_transforms(self):
        """infoïRGBïinfo"""
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
        
        # info
        if self.normalize_images:
            if self.custom_image_norm_stats:
                mean = self.custom_image_norm_stats["mean"]
                std = self.custom_image_norm_stats["std"]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            else:
                # ImageNet pretrained weights
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
        print(f"info: RGB, info: {self.normalize_images}")
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        """info"""
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
        """RGB"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                return torch.zeros(3, 224, 224)
            
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"info {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_sequence_data(self, json_path):
        """info"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            original_length = len(frames)
            
            # info
            if len(frames) < self.sequence_length:
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[-self.sequence_length:]
            
            # info
            joints_list = []  # joint1joint6
            labels_list = []
            images_list = []
            
            for frame in frames:
                # info
                all_joints = frame.get('joints', [0.0] * 7)
                all_joints = [float(j) if j is not None else 0.0 for j in all_joints]
                
                # joint1(0)joint6(Step 5)
                if len(all_joints) >= 6:
                    selected_joints = [all_joints[0], all_joints[5]]  # joint1, joint6
                else:
                    selected_joints = [0.0, 0.0]
                
                joints_list.append(selected_joints)
                
                # info
                label = frame.get('label', 0)
                labels_list.append(float(label) if label is not None else 0.0)
                
                # info
                image_path = frame.get('image_path', '')
                
                if image_path:
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                    else:
                        relative_image_path = image_path
                    
                    # info
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                    
                    image = self._load_image(relative_image_path)
                else:
                    image = torch.zeros(3, 224, 224)
                
                images_list.append(image)
            
            # info
            joints = torch.tensor(joints_list, dtype=torch.float32)  # shape: [seq_len, 2]
            labels = torch.tensor(labels_list, dtype=torch.float32)
            images = torch.stack(images_list)
            
            # shuffle_jointsïjointsïimagesïinfo
            if self.shuffle_joints:
                # info
                shuffle_indices = torch.randperm(len(joints))
                joints = joints[shuffle_indices]
            
            ball_count = json_data.get('ball_count', 0)
            
            return {
                'joints': joints,  # joint1joint6ïinfo
                'labels': labels,
                'images': images,
                'sequence_length': original_length,
                'ball_count': int(ball_count)
            }
            
        except Exception as e:
            print(f"JSON {json_path}: {e}")
            import traceback
            traceback.print_exc()
            
            # info
            return {
                'joints': torch.zeros(self.sequence_length, 2, dtype=torch.float32),  # 2joint
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
    info
    
    Args:
        train_csv_path: CSV
            - "ball_counting_dataset_train.csv" Step 100%info
            - "ball_counting_dataset_train_10.csv" Step 10%info
            - "ball_counting_dataset_train_50.csv" Step 50%info
        val_csv_path: CSV
        data_root: info
        batch_size: info
        sequence_length: info
        num_workers: info
        normalize_images: infoïImageNet pretrained weightsïinfo
        custom_image_norm_stats: info {"mean": [...], "std": [...]}
        shuffle_joints: jointsïinfoïinfo
        curriculum_mode: info
            - 'random': info
            - 'easy_to_hard': infoïinfoïinfo
            - 'hard_to_easy': infoïinfoïinfo
        seed: info
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== info - RGB ===")
    print("=" * 60)
    
    print("\n[info]")
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
    
    print("\n[info]")
    val_dataset = BallCountingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        shuffle_joints=False,  # joints
        curriculum_mode='random',  # info
        seed=seed
    )
    
    # info: shuffle=Trueïcurriculum_mode
    # curriculum_mode='random'ïinfoïeasy_to_hardhard_to_easyïinfo
    shuffle_train = (curriculum_mode == 'random')
    
    # pin_memoryCUDAïGPUïinfo
    # infoïTrue
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
    print(f"info: {len(train_dataset)} info, Batch shuffle: {shuffle_train}")
    print(f"info: {len(val_dataset)} info")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# info
if __name__ == "__main__":
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv_100 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    train_csv_50 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_50.csv"
    train_csv_10 = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train_10.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== infoïinfo + infoïinfo===\n")
    
    if not os.path.exists(train_csv_100):
        print(f"info: CSV: {train_csv_100}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"info: CSV: {val_csv}")
        exit(1)
    
    try:
        # ============ Step 1: infoïStep 100%infoïinfoïjointsïinfo============
        print("\n" + "="*70)
        print("Step 1: info - 100%info")
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
        
        # ============ Step 2: Joint - 100%info ============
        print("\n" + "="*70)
        print("Step 2: Joint - 100%info")
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
        
        # ============ Step 3: info - 100%info ============
        print("\n" + "="*70)
        print("Step 3: info - info - 100%info")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='easy_to_hard'
        )
        
        # ============ Step 4: info - 100%info ============
        print("\n" + "="*70)
        print("Step 4: info - info - 100%info")
        print("="*70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='hard_to_easy'
        )
        
        # ============ Step 5: Step 10%info ============
        print("\n" + "="*70)
        print("Step 5: Step 10%info")
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
            print(f"10%CSV: {train_csv_10}")
        
        # ============ Step 6: Step 50%info ============
        print("\n" + "="*70)
        print("Step 6: Step 50%info")
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
            print(f"50%CSV: {train_csv_50}")
        
        # ============ Step 7: info - Joint + info + 10%info ============
        print("\n" + "="*70)
        print("Step 7: infoïJoint + info + 10%infoïinfo")
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
        print("infoïinfo")
        print("="*70)
        
        print("\n:")
        print("-" * 70)
        print("# 1. infoïStep 100%infoïinfo")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root)")
        print()
        print("# 2. Step 10%info")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root)")
        print()
        print("# 3. Step 50%info")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root)")
        print()
        print("# 4. JointïStep 100%infoïinfo")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root,")
        print("    shuffle_joints=True)")
        print()
        print("# 5. infoïStep 50%infoïinfo")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root,")
        print("    curriculum_mode='easy_to_hard')")
        print()
        print("# 6. infoïJoint + info + 10%info")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root,")
        print("    shuffle_joints=True,")
        print("    curriculum_mode='easy_to_hard')")
        print("-" * 70)
        
    except Exception as e:
        print(f"info: {e}")
        import traceback
        traceback.print_exc()

