import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SingleImageDataset(Dataset):
    """info - infoïinfoïinfo"""
    
    def __init__(self, csv_path, data_root, 
                 sequence_length=11,
                 normalize_images=True, 
                 custom_image_norm_stats=None):
        """
        info
        
        Args:
            csv_path: CSV
            data_root: info
            sequence_length: infoïinfoïinfo
            normalize_images: infoïImageNet pretrained weightsïinfo
            custom_image_norm_stats: info {"mean": [...], "std": [...]}
        """
        # CSV
        csv_data = pd.read_csv(csv_path)
        
        # CSV
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV{col}info"
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # info
        self._setup_image_transforms()
        
        # info
        self.samples = self._build_sample_list()
        
        print(f"info:")
        print(f"  info: {len(self.csv_data)}")
        print(f"  info: {len(self.samples)}")
        print(f"  info: RGB")
        print(f"  info: ball_count (1-10)")
    
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
                # ImageNet
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
        print(f"info: RGB, info: {self.normalize_images}")
    
    def _build_sample_list(self):
        """info - infoïinfoïinfo"""
        samples = []
        
        for idx in range(len(self.csv_data)):
            sample_row = self.csv_data.iloc[idx]
            sample_id = sample_row['sample_id']
            ball_count = sample_row['ball_count']
            json_path = sample_row['json_path']
            
            try:
                # JSON
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                frames = json_data['frames']
                original_length = len(frames)
                
                # infoïembodimentïinfo
                if len(frames) < self.sequence_length:
                    # infoïinfo
                    last_frame = frames[-1]
                    frames = frames + [last_frame] * (self.sequence_length - len(frames))
                elif len(frames) > self.sequence_length:
                    frames = frames[:self.sequence_length]
                
                # infoïinfoïinfo
                for frame_idx, frame in enumerate(frames):
                    # info
                    image_path = frame.get('image_path', '')
                    if image_path:
                        # info
                        path_parts = image_path.split('/')
                        if 'ball_data_collection' in path_parts:
                            ball_data_idx = path_parts.index('ball_data_collection')
                            relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                        else:
                            relative_image_path = image_path
                        
                        # info
                        if '1_ball' in relative_image_path:
                            relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                        
                        # info
                        sample = {
                            'image_path': relative_image_path,
                            'label': int(ball_count),
                            'sample_id': sample_id,
                            'frame_idx': frame_idx
                        }
                        samples.append(sample)
                
            except Exception as e:
                print(f"JSON {json_path}: {e}")
                continue
        
        return samples
    
    def _load_image(self, image_path):
        """RGB"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"info: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            # info
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"info {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """info"""
        sample = self.samples[idx]
        
        # info
        image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'label': sample['label'],
            'sample_id': sample['sample_id'],
            'frame_idx': sample['frame_idx']
        }
    
    def get_class_distribution(self):
        """info"""
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
    info
    
    Args:
        train_csv_path: CSV
        val_csv_path: CSV
        data_root: info
        batch_size: info
        sequence_length: infoïinfoïinfo
        num_workers: info
        normalize_images: infoïImageNet pretrained weightsïinfo
        custom_image_norm_stats: info {"mean": [...], "std": [...]}
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== info - RGB ===")
    print("=" * 60)
    
    print("=" * 60)
    print("=== info - RGB ===")
    print("=" * 60)
    
    print("\n[info]")
    # info
    train_dataset = SingleImageDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[info]")
    # info
    val_dataset = SingleImageDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # info
    print("\n:")
    train_dist = train_dataset.get_class_distribution()
    for label in sorted(train_dist.keys()):
        print(f"  info {label}: {train_dist[label]} info")
    
    print("\n:")
    val_dist = val_dataset.get_class_distribution()
    for label in sorted(val_dist.keys()):
        print(f"  info {label}: {val_dist[label]} info")
    
    # pin_memoryCUDA
    use_pin_memory = True
    
    # info
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
    print(f"info: {len(train_dataset)} info")
    print(f"info: {len(val_dataset)} info")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# info
if __name__ == "__main__":
    # info
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== info ===")
    
    # info
    if not os.path.exists(train_csv):
        print(f"info: CSV: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"info: CSV: {val_csv}")
        exit(1)
    if not os.path.exists(data_root):
        print(f"info: info: {data_root}")
        exit(1)
    
    try:
        # info
        train_loader, val_loader = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize_images=True
        )
        
        print(f"info: {len(train_loader.dataset)}")
        print(f"info: {len(val_loader.dataset)}")
        
        # batch
        for batch in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Labels: {batch['label'].shape}")
            print(f"  info: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"  info: [{batch['label'].min()}, {batch['label'].max()}]")
            print(f"  info: {batch['label'][:5].tolist()}")
            break
        
        print("\n=== info ===")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=32, sequence_length=11)")
        
        print("\n=== info ===")
        
    except Exception as e:
        print(f"info: {e}")
        import traceback
        traceback.print_exc()