import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SingleImageDataset(Dataset):
    """Single-image dataset — each frame is treated as an independent sample."""
    
    def __init__(self, csv_path, data_root, 
                 sequence_length=11,
                 normalize_images=True, 
                 custom_image_norm_stats=None):
        """
        Initialize the single-image dataset.
        
        Args:
            csv_path: Path to the CSV annotation file.
            data_root: Root directory of image data.
            sequence_length: Fixed temporal length to pad/truncate each sequence to.
            normalize_images: Whether to normalize images (ImageNet stats by default).
            custom_image_norm_stats: Custom normalization params, format: {"mean": [...], "std": [...]}.
        """
        # Load CSV
        csv_data = pd.read_csv(csv_path)
        
        # Validate required CSV columns
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV missing required column: {col}"
        
        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        
        # Initialize image transforms
        self._setup_image_transforms()
        
        # Build the flat list of individual frame samples
        self.samples = self._build_sample_list()
        
        print(f"Dataset summary:")
        print(f"  Sequences: {len(self.csv_data)}")
        print(f"  Total frames: {len(self.samples)}")
        print(f"  Image mode: RGB")
        print(f"  Label range: ball_count (1-10)")
    
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
    
    def _build_sample_list(self):
        """Build a flat list of individual frame samples from all sequences."""
        samples = []
        
        for idx in range(len(self.csv_data)):
            sample_row = self.csv_data.iloc[idx]
            sample_id = sample_row['sample_id']
            ball_count = sample_row['ball_count']
            json_path = sample_row['json_path']
            
            try:
                # Load JSON sequence file
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                frames = json_data['frames']
                original_length = len(frames)
                
                # Pad or truncate to the fixed sequence length
                if len(frames) < self.sequence_length:
                    # Repeat the last frame to fill the gap
                    last_frame = frames[-1]
                    frames = frames + [last_frame] * (self.sequence_length - len(frames))
                elif len(frames) > self.sequence_length:
                    frames = frames[:self.sequence_length]
                
                # Register each frame as an independent sample
                for frame_idx, frame in enumerate(frames):
                    # Get the image path for this frame
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
                        
                        # Append frame entry to the sample list
                        sample = {
                            'image_path': relative_image_path,
                            'label': int(ball_count),
                            'sample_id': sample_id,
                            'frame_idx': frame_idx
                        }
                        samples.append(sample)
                
            except Exception as e:
                print(f"Failed to read JSON {json_path}: {e}")
                continue
        
        return samples
    
    def _load_image(self, image_path):
        """Load a single image and convert it to an RGB Tensor."""
        try:
            full_image_path = os.path.join(self.data_root, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found: {full_image_path}")
                return torch.zeros(3, 224, 224)
            
            # Open and apply transforms
            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)
            
            return image
            
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Retrieve a single frame sample."""
        sample = self.samples[idx]
        
        # Load and transform the image
        image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'label': sample['label'],
            'sample_id': sample['sample_id'],
            'frame_idx': sample['frame_idx']
        }
    
    def get_class_distribution(self):
        """Return a dict mapping each ball_count label to its frame count."""
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
    Build training and validation DataLoaders for single-image classification.
    
    Args:
        train_csv_path: Path to the training CSV file.
        val_csv_path: Path to the validation CSV file.
        data_root: Root directory of image data.
        batch_size: Batch size.
        sequence_length: Fixed temporal length to pad/truncate each sequence to.
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
    # Build training dataset
    train_dataset = SingleImageDataset(
        csv_path=train_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    print("\n[Validation Set]")
    # Build validation dataset
    val_dataset = SingleImageDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats
    )
    
    # Print per-class frame counts for both splits
    print("\nTraining class distribution:")
    train_dist = train_dataset.get_class_distribution()
    for label in sorted(train_dist.keys()):
        print(f"  Ball count {label}: {train_dist[label]} frames")
    
    print("\nValidation class distribution:")
    val_dist = val_dataset.get_class_distribution()
    for label in sorted(val_dist.keys()):
        print(f"  Ball count {label}: {val_dist[label]} frames")
    
    # pin_memory is beneficial for CUDA training and safe to enable on CPU too.
    use_pin_memory = True
    
    # Build DataLoaders
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
    print(f"Training set: {len(train_dataset)} frames")
    print(f"Validation set: {len(val_dataset)} frames")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader


# Usage example
if __name__ == "__main__":
    # Configure paths
    data_root = "data/ball_data_collection"
    train_csv = "data/Tools_script/ball_counting_dataset_train.csv"
    val_csv = "data/Tools_script/ball_counting_dataset_val.csv"
   
    print("=== SingleImageDataset Example ===")
    
    # Validate paths before proceeding
    if not os.path.exists(train_csv):
        print(f"Error: training CSV not found: {train_csv}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"Error: validation CSV not found: {val_csv}")
        exit(1)
    if not os.path.exists(data_root):
        print(f"Error: data root not found: {data_root}")
        exit(1)
    
    try:
        # Build DataLoaders
        train_loader, val_loader = get_single_image_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            normalize_images=True
        )
        
        print(f"Training frames: {len(train_loader.dataset)}")
        print(f"Validation frames: {len(val_loader.dataset)}")
        
        # Inspect the first batch
        for batch in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Labels: {batch['label'].shape}")
            print(f"  Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"  Label range: [{batch['label'].min()}, {batch['label'].max()}]")
            print(f"  Sample labels: {batch['label'][:5].tolist()}")
            break
        
        print("\n=== Usage Example ===")
        print("train_loader, val_loader = get_single_image_data_loaders(")
        print("    train_csv, val_csv, data_root,")
        print("    batch_size=32, sequence_length=11)")
        
        print("\n=== Done ===")
        
    except Exception as e:
        print(f"Run failed: {e}")
        import traceback
        traceback.print_exc()