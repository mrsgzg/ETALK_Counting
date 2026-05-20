import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class BallCountingDataset(Dataset):

    def __init__(self, csv_path, data_root, sequence_length=6,
                 normalize_images=True, custom_image_norm_stats=None,
                 shuffle_joints=False, curriculum_mode='random', seed=42):
        """
        Initialize the ball-counting sequence dataset.

        Args:
            csv_path: Path to the CSV annotation file.
            data_root: Root directory of image data.
            sequence_length: Fixed temporal length to pad/truncate each sample to.
            normalize_images: Whether to normalize images (ImageNet stats by default).
            custom_image_norm_stats: Custom normalization params, format: {"mean": [...], "std": [...]}.
            shuffle_joints: Whether to shuffle the joints sequence within a sample (for ablation studies).
            curriculum_mode: Sample ordering mode.
                - 'random': Randomly shuffle training samples.
                - 'easy_to_hard': Sort by ball_count ascending.
                - 'hard_to_easy': Sort by ball_count descending.
            seed: Random seed.
        """
        # Load CSV
        csv_data = pd.read_csv(csv_path)

        # Validate required CSV columns
        required_columns = ['sample_id', 'ball_count', 'json_path']
        for col in required_columns:
            assert col in csv_data.columns, f"CSV missing required column: {col}"

        print(f"Dataset size: {len(csv_data)} samples")

        # Reorder data according to curriculum mode
        if curriculum_mode == 'easy_to_hard':
            csv_data = csv_data.sort_values('ball_count', ascending=True).reset_index(drop=True)
            print("Curriculum mode: easy_to_hard")
        elif curriculum_mode == 'hard_to_easy':
            csv_data = csv_data.sort_values('ball_count', ascending=False).reset_index(drop=True)
            print("Curriculum mode: hard_to_easy")
        elif curriculum_mode == 'random':
            # Random order
            csv_data = csv_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            print("Curriculum mode: random")
        else:
            raise ValueError(
                f"Unsupported curriculum_mode: {curriculum_mode}. "
                "Valid options are 'random', 'easy_to_hard', 'hard_to_easy'"
            )

        self.csv_data = csv_data
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize_images = normalize_images
        self.custom_image_norm_stats = custom_image_norm_stats
        self.shuffle_joints = shuffle_joints
        self.seed = seed

        if self.shuffle_joints:
            print("Joint mode: enabled (joints sequence will be shuffled)")
        else:
            print("Joint mode: disabled (original joints sequence preserved)")

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
            'sequence_data': sequence_data,
            'label': ball_count
        }

    def _load_image(self, image_path):
        """Load a single image and convert it to an RGB Tensor."""
        try:
            full_image_path = os.path.join(self.data_root, image_path)

            if not os.path.exists(full_image_path):
                return torch.zeros(3, 224, 224)

            image = Image.open(full_image_path).convert('RGB')
            image = self.image_transform(image)

            return image

        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, 224, 224)

    def _load_sequence_data(self, json_path):
        """Read a JSON file and assemble a temporal sequence sample."""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            frames = json_data['frames']
            original_length = len(frames)

            # Pad or truncate to the fixed sequence length
            if len(frames) < self.sequence_length:
                last_frame = frames[-1]
                frames = frames + [last_frame] * (self.sequence_length - len(frames))
            elif len(frames) > self.sequence_length:
                frames = frames[-self.sequence_length:]

            # Collect per-modality sequences
            joints_list = []  # Keep only joint1 (index 0) and joint6 (index 5)
            labels_list = []
            images_list = []

            for frame in frames:
                # Read and clean joints
                all_joints = frame.get('joints', [0.0] * 7)
                all_joints = [float(j) if j is not None else 0.0 for j in all_joints]

                # Select joint1 (index 0) and joint6 (index 5) only
                if len(all_joints) >= 6:
                    selected_joints = [all_joints[0], all_joints[5]]
                else:
                    selected_joints = [0.0, 0.0]

                joints_list.append(selected_joints)

                # Read per-frame label
                label = frame.get('label', 0)
                labels_list.append(float(label) if label is not None else 0.0)

                # Read image
                image_path = frame.get('image_path', '')

                if image_path:
                    path_parts = image_path.split('/')
                    if 'ball_data_collection' in path_parts:
                        ball_data_idx = path_parts.index('ball_data_collection')
                        relative_image_path = '/'.join(path_parts[ball_data_idx + 1:])
                    else:
                        relative_image_path = image_path

                    # Handle legacy path naming: 1_ball -> 1_balls
                    if '1_ball' in relative_image_path:
                        relative_image_path = relative_image_path.replace('1_ball', '1_balls')

                    image = self._load_image(relative_image_path)
                else:
                    image = torch.zeros(3, 224, 224)

                images_list.append(image)

            # Convert to Tensors
            joints = torch.tensor(joints_list, dtype=torch.float32)  # [seq_len, 2]
            labels = torch.tensor(labels_list, dtype=torch.float32)
            images = torch.stack(images_list)

            # Shuffle joints only, keeping image order intact (to break cross-modal alignment)
            if self.shuffle_joints:
                shuffle_indices = torch.randperm(len(joints))
                joints = joints[shuffle_indices]

            ball_count = json_data.get('ball_count', 0)

            return {
                'joints': joints,
                'labels': labels,
                'images': images,
                'sequence_length': original_length,
                'ball_count': int(ball_count)
            }

        except Exception as e:
            print(f"Failed to read JSON {json_path}: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: return zero tensors to prevent DataLoader from crashing
            return {
                'joints': torch.zeros(self.sequence_length, 2, dtype=torch.float32),
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
    Build training and validation DataLoaders.

    Args:
        train_csv_path: Path to the training CSV file.
            - ball_counting_dataset_train.csv: 100% training data
            - ball_counting_dataset_train_10.csv: 10% training data
            - ball_counting_dataset_train_50.csv: 50% training data
        val_csv_path: Path to the validation CSV file.
        data_root: Root directory of image data.
        batch_size: Batch size.
        sequence_length: Sequence length.
        num_workers: Number of DataLoader worker processes.
        normalize_images: Whether to normalize images (ImageNet stats by default).
        custom_image_norm_stats: Custom normalization params {"mean": [...], "std": [...]}.
        shuffle_joints: Whether to shuffle the joints sequence.
        curriculum_mode: Sample ordering mode.
            - 'random': Random order
            - 'easy_to_hard': Sort ascending by difficulty
            - 'hard_to_easy': Sort descending by difficulty
        seed: Random seed.

    Returns:
        train_loader, val_loader
    """

    print("=" * 60)
    print("=== Creating DataLoaders (RGB) ===")
    print("=" * 60)

    print("\n[Training Set]")
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

    print("\n[Validation Set]")
    val_dataset = BallCountingDataset(
        csv_path=val_csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        normalize_images=normalize_images,
        custom_image_norm_stats=custom_image_norm_stats,
        shuffle_joints=False,  # Always preserve original joints for validation
        curriculum_mode='random',  # Fixed random order for validation
        seed=seed
    )

    # Enable DataLoader shuffle only in random mode,
    # to avoid conflicting with easy_to_hard / hard_to_easy curriculum ordering.
    shuffle_train = (curriculum_mode == 'random')

    # pin_memory is beneficial for CUDA training and safe to enable on CPU too.
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
    print(f"Training set: {len(train_dataset)} samples, DataLoader shuffle={shuffle_train}")
    print(f"Validation set: {len(val_dataset)} samples")
    print("=" * 60 + "\n")

    return train_loader, val_loader


# Usage examples
if __name__ == "__main__":
    data_root = "data/ball_data_collection"
    train_csv_100 = "data/Tools_script/ball_counting_dataset_train.csv"
    train_csv_50 = "data/Tools_script/ball_counting_dataset_train_50.csv"
    train_csv_10 = "data/Tools_script/ball_counting_dataset_train_10.csv"
    val_csv = "data/Tools_script/ball_counting_dataset_val.csv"

    print("=== DataLoader Examples: Various Training Configurations ===\n")

    if not os.path.exists(train_csv_100):
        print(f"Error: training CSV not found: {train_csv_100}")
        exit(1)
    if not os.path.exists(val_csv):
        print(f"Error: validation CSV not found: {val_csv}")
        exit(1)

    try:
        # ============ Step 1: Baseline (100% data, joints not shuffled) ============
        print("\n" + "=" * 70)
        print("Step 1: Baseline - 100% data")
        print("=" * 70)
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

        # ============ Step 2: Joint ablation (100% data) ============
        print("\n" + "=" * 70)
        print("Step 2: Joint Ablation - 100% data")
        print("=" * 70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            shuffle_joints=True,
            curriculum_mode='random'
        )

        # ============ Step 3: Curriculum learning (easy to hard, 100% data) ============
        print("\n" + "=" * 70)
        print("Step 3: Curriculum Learning - Easy to Hard - 100% data")
        print("=" * 70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='easy_to_hard'
        )

        # ============ Step 4: Curriculum learning (hard to easy, 100% data) ============
        print("\n" + "=" * 70)
        print("Step 4: Curriculum Learning - Hard to Easy - 100% data")
        print("=" * 70)
        train_loader, val_loader = get_ball_counting_data_loaders(
            train_csv_path=train_csv_100,
            val_csv_path=val_csv,
            data_root=data_root,
            batch_size=16,
            sequence_length=11,
            curriculum_mode='hard_to_easy'
        )

        # ============ Step 5: 10% data ============
        print("\n" + "=" * 70)
        print("Step 5: 10% Training Data")
        print("=" * 70)
        if os.path.exists(train_csv_10):
            train_loader, val_loader = get_ball_counting_data_loaders(
                train_csv_path=train_csv_10,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                sequence_length=11
            )
        else:
            print(f"10% CSV not found: {train_csv_10}")

        # ============ Step 6: 50% data ============
        print("\n" + "=" * 70)
        print("Step 6: 50% Training Data")
        print("=" * 70)
        if os.path.exists(train_csv_50):
            train_loader, val_loader = get_ball_counting_data_loaders(
                train_csv_path=train_csv_50,
                val_csv_path=val_csv,
                data_root=data_root,
                batch_size=16,
                sequence_length=11
            )
        else:
            print(f"50% CSV not found: {train_csv_50}")

        # ============ Step 7: Combined config (Joint + curriculum + 10% data) ============
        print("\n" + "=" * 70)
        print("Step 7: Combined Config - Joint + Easy to Hard + 10% data")
        print("=" * 70)
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

        print("\n" + "=" * 70)
        print("Example run complete")
        print("=" * 70)

        print("\nCommon usage examples:")
        print("-" * 70)
        print("# 1. Baseline: 100% data")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root)")
        print()
        print("# 2. 10% training data")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root)")
        print()
        print("# 3. 50% training data")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root)")
        print()
        print("# 4. Joint ablation: 100% data")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train.csv', val_csv, data_root,")
        print("    shuffle_joints=True)")
        print()
        print("# 5. Curriculum learning: 50% data (easy to hard)")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_50.csv', val_csv, data_root,")
        print("    curriculum_mode='easy_to_hard')")
        print()
        print("# 6. Combined config: Joint + curriculum + 10% data")
        print("train_loader, val_loader = get_ball_counting_data_loaders(")
        print("    'ball_counting_dataset_train_10.csv', val_csv, data_root,")
        print("    shuffle_joints=True,")
        print("    curriculum_mode='easy_to_hard')")
        print("-" * 70)

    except Exception as e:
        print(f"Run failed: {e}")
        import traceback
        traceback.print_exc()