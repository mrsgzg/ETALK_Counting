import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SequencePoolingDataset(Dataset):
    """info - infoïjointsïinfo"""
    
    def __init__(self, csv_path, data_root, sequence_length=11, 
                 normalize_images=True, custom_image_norm_stats=None):
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
        
        print(f"info: {len(csv_data)} info")
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
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
                # ImageNet
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
            'images': sequence_data,  # [seq_len, C, H, W]
            'label': ball_count
        }
    
    def _load_image(self, image_path):
        """RGB"""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"info: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"info {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_sequence_data(self, json_path):
        """infoïinfoïinfo"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            
            # infoïembodimentïinfo
            if len(frames) < self.sequence_length:
                # infoïinfo
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[:self.sequence_length]
            
            # info
            images_list = []
            for frame in frames:
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
                    
                    image = self._load_image(relative_image_path)
                    images_list.append(image)
                else:
                    # infoïinfo
                    images_list.append(torch.zeros(3, 224, 224))
            
            # info [seq_len, C, H, W]
            images_tensor = torch.stack(images_list, dim=0)
            
            return images_tensor
            
        except Exception as e:
            print(f"info {json_path}: {e}")
            # info
            return torch.zeros(self.sequence_length, 3, 224, 224)


def get_sequence_pooling_data_loaders(train_csv_path, val_csv_path, data_root, 
                                      batch_size=16, sequence_length=11, 
                                      num_workers=1, normalize_images=True,
                                      custom_image_norm_stats=None):
    """
    info
    
    Args:
        train_csv_path: CSV
        val_csv_path: CSV
        data_root: info
        batch_size: info
        sequence_length: info
        num_workers: info
        normalize_images: infoïImageNet pretrained weightsïinfo
        custom_image_norm_stats: info {"mean": [...], "std": [...]}
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== info - RGB ===")
    print("=" * 60)
    
    print("\n[info]")
    train_dataset = SequencePoolingDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[info]")
    val_dataset = SequencePoolingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # pin_memoryCUDA
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
    print(f"info: {len(train_dataset)} info")
    print(f"info: {len(val_dataset)} info")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# info
if __name__ == "__main__":
    data_root = "/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection"
    train_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== info ===\n")
    
    if not os.path.exists(train_csv):
        print(f"info: CSV: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"info: CSV: {val_csv}")
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
        
        # batch
        for batch in train_loader:
            print(f"Batch shapes:")
            print(f"  Images: {batch['images'].shape}")  # [B, 11, C, H, W]
            print(f"  Labels: {batch['label'].shape}")
            print(f"  info: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
            print(f"  info: [{batch['label'].min()}, {batch['label'].max()}]")
            break
        
        print("\n=== info ===")
        print("train_loader, val_loader = get_sequence_pooling_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=16, sequence_length=11)")
        
        print("\n=== info ===")
        
    except Exception as e:
        print(f"info: {e}")
        import traceback
        traceback.print_exc()
