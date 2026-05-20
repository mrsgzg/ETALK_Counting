import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SequencePoolingDataset(Dataset):
    """Image-only sequence dataset — joints are not used."""
    
    def __init__(self, csv_path, data_root, sequence_length=11, 
                 normalize_images=True, custom_image_norm_stats=None):
        """
        Initialize the sequence pooling dataset.
        
        Args:
            csv_path: Path to the CSV annotation file.
            data_root: Root directory of image data.
            sequence_length: Fixed temporal length to pad/truncate each sample to.
            normalize_images: Whether to normalize images (ImageNet stats by default).
            custom_image_norm_stats: Custom normalization params, format: {"mean": [...], "std": [...]}.
        """
        # Load CSV
        csv_data = pd.read_csv(csv_path)
        
        # Validate required CSV columns
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV missing required column: {col}"
        
        print(f"Dataset size: {len(csv_data)} samples")
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # Initialize image transforms
        self._setup_image_transforms()
    
    def _setup_image_transforms(self):
        """Build the RGB image preprocessing pipeline."""
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
        
        # Optional normalization
        if self.normalize_images:
            if self.custom_image_norm_stats:
                mean = self.custom_image_norm_stats["mean"]
                std = self.custom_image_norm_stats["std"]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            else:
                # Default to ImageNet pre-training statistics
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
        
        self.image_transform = transforms.Compose(transform_list)
        print(f"Image preprocessing: RGB, normalize={self.normalize_images}")
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        """Retrieve a single sample."""
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
        """Load a single image and convert it to an RGB Tensor."""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_sequence_data(self, json_path):
        """Read a JSON file and assemble an image sequence sample."""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            frames = json_data['frames']
            
            # Pad or truncate to the fixed sequence length
            if len(frames) < self.sequence_length:
                # Repeat the last frame to fill the gap
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[:self.sequence_length]
            
            # Load images for each frame
            images_list = []
            for frame in frames:
                image_path = frame.get('image_path', '')
                if image_path:
                    # Strip the absolute prefix and keep the relative path
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx+1:])
                    else:
                        relative_image_path = image_path
                    
                    # Handle legacy path naming: 1_ball -> 1_balls
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')
                    
                    image = self._load_image(relative_image_path)
                    images_list.append(image)
                else:
                    # No image path provided — use a zero tensor as placeholder
                    images_list.append(torch.zeros(3, 224, 224))
            
            # Stack into a single tensor: [seq_len, C, H, W]
            images_tensor = torch.stack(images_list, dim=0)
            
            return images_tensor
            
        except Exception as e:
            print(f"Failed to read JSON {json_path}: {e}")
            # Fallback: return zero tensor to prevent DataLoader from crashing
            return torch.zeros(self.sequence_length, 3, 224, 224)


def get_sequence_pooling_data_loaders(train_csv_path, val_csv_path, data_root, 
                                      batch_size=16, sequence_length=11, 
                                      num_workers=1, normalize_images=True,
                                      custom_image_norm_stats=None):
    """
    Build training and validation DataLoaders.
    
    Args:
        train_csv_path: Path to the training CSV file.
        val_csv_path: Path to the validation CSV file.
        data_root: Root directory of image data.
        batch_size: Batch size.
        sequence_length: Sequence length.
        num_workers: Number of DataLoader worker processes.
        normalize_images: Whether to normalize images (ImageNet stats by default).
        custom_image_norm_stats: Custom normalization params {"mean": [...], "std": [...]}.
    
    Returns:
        train_loader, val_loader
    """
    
    print("=" * 60)
    print("=== Creating DataLoaders (RGB) ===")
    print("=" * 60)
    
    print("\n[Training Set]")
    train_dataset = SequencePoolingDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[Validation Set]")
    val_dataset = SequencePoolingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # pin_memory is beneficial for CUDA training and safe to enable on CPU too.
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
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# Usage example
if __name__ == "__main__":
    data_root = "data/ball_data_collection"
    train_csv = "data/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "data/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== DataLoader Example ===\n")
    
    if not os.path.exists(train_csv):
        print(f"Error: training CSV not found: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"Error: validation CSV not found: {val_csv}")
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
        
        # Inspect the first batch
        for batch in train_loader:
            print(f"Batch shapes:")
            print(f"  Images: {batch['images'].shape}")  # [B, 11, C, H, W]
            print(f"  Labels: {batch['label'].shape}")
            print(f"  Image value range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
            print(f"  Label range: [{batch['label'].min()}, {batch['label'].max()}]")
            break
        
        print("\n=== Usage Example ===")
        print("train_loader, val_loader = get_sequence_pooling_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=16, sequence_length=11)")
        
        print("\n=== Done ===")
        
    except Exception as e:
        print(f"Run failed: {e}")
        import traceback
        traceback.print_exc()