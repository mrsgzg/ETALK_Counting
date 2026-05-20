import os
import json
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def collect_ball_samples(root_dir, output_dir="./"):
    """
    Collect all ball-counting samples and generate CSV files.
    
    Args:
        root_dir (str): Root directory containing 1_balls, 2_balls, ..., 10_balls folders
        output_dir (str): Directory to save the output CSV files
    """
    
    # Accumulator for all sample metadata
    samples = []
    
    # Iterate over 1_balls through 10_balls folders
    for ball_count in range(1, 11):
        folder_name = f"{ball_count}_balls"
        folder_path = Path(root_dir) / folder_name / "metadata"
        
        print(f"Processing folder: {folder_name}")
        
        # Skip if folder does not exist
        if not folder_path.exists():
            print(f"Warning: folder {folder_path} not found — skipping.")
            continue
            
        # Find all JSON files in the metadata folder
        json_files = list(folder_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                # Load JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract relevant fields
                sample_info = {
                    'json_path': str(json_file),
                    'sample_id': data.get('sample_id', ''),
                    'ball_count': data.get('ball_count', ball_count),
                    'sequence_length': data.get('sequence_length', len(data.get('frames', []))),
                    'collection_time': data.get('collection_time', ''),
                    'folder_name': folder_name,
                    'relative_path': str(json_file.relative_to(root_dir))
                }
                
                # Warn if ball_count in JSON does not match the folder name
                if sample_info['ball_count'] != ball_count:
                    print(f"Warning: ball_count in {json_file} ({sample_info['ball_count']}) does not match folder ({ball_count})")
                
                samples.append(sample_info)
                
            except Exception as e:
                print(f"Error: could not process {json_file}: {e}")
                continue
    
    # Build DataFrame
    df = pd.DataFrame(samples)
    
    if df.empty:
        print("Error: no valid samples found.")
        return
    
    print(f"\nTotal samples collected: {len(df)}")
    print(f"Ball count distribution:")
    print(df['ball_count'].value_counts().sort_index())
    
    # Save the full dataset CSV
    total_csv_path = Path(output_dir) / "ball_counting_dataset_all.csv"
    df.to_csv(total_csv_path, index=False, encoding='utf-8')
    print(f"\nSaved full dataset to: {total_csv_path}")
    
    # Stratified train/val split (80% train, 20% val) per ball count
    train_samples = []
    val_samples = []
    
    print(f"\n=== Splitting dataset by ball count ===")
    for ball_count in sorted(df['ball_count'].unique()):
        ball_samples = df[df['ball_count'] == ball_count]
        total_count = len(ball_samples)
        
        print(f"Ball count {ball_count}: {total_count} samples", end=" -> ")
        
        if total_count == 1:
            # Only one sample — put it in the training set
            train_samples.append(ball_samples)
            print("train: 1, val: 0 (too few samples)")
        elif total_count == 2:
            # Two samples — one in train, one in val
            train_part, val_part = train_test_split(
                ball_samples, 
                test_size=0.5, 
                random_state=42, 
                shuffle=True
            )
            train_samples.append(train_part)
            val_samples.append(val_part)
            print("train: 1, val: 1 (minimum split)")
        else:
            # Standard 80/20 stratified split
            train_part, val_part = train_test_split(
                ball_samples, 
                test_size=0.2, 
                random_state=42, 
                shuffle=True
            )
            train_samples.append(train_part)
            val_samples.append(val_part)
            print(f"train: {len(train_part)}, val: {len(val_part)} "
                  f"(ratio: {len(train_part)/(len(train_part)+len(val_part))*100:.1f}%"
                  f":{len(val_part)/(len(train_part)+len(val_part))*100:.1f}%)")
    
    # Concatenate splits
    train_df = pd.concat(train_samples, ignore_index=True) if train_samples else pd.DataFrame()
    val_df = pd.concat(val_samples, ignore_index=True) if val_samples else pd.DataFrame()
    
    # Save training CSV
    train_csv_path = Path(output_dir) / "ball_counting_dataset_train.csv"
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    print(f"Saved training set to: {train_csv_path} ({len(train_df)} samples)")
    
    # Save validation CSV
    val_csv_path = Path(output_dir) / "ball_counting_dataset_val.csv"
    val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    print(f"Saved validation set to: {val_csv_path} ({len(val_df)} samples)")
    
    # Print dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    print(f"\nTraining set ball count distribution:")
    if not train_df.empty:
        train_counts = train_df['ball_count'].value_counts().sort_index()
        print(train_counts)
    
    print(f"\nValidation set ball count distribution:")
    if not val_df.empty:
        val_counts = val_df['ball_count'].value_counts().sort_index()
        print(val_counts)
    
    # Per-class split ratio check
    print(f"\n=== Per-class split ratio check ===")
    for ball_count in sorted(df['ball_count'].unique()):
        total_count = len(df[df['ball_count'] == ball_count])
        train_count = len(train_df[train_df['ball_count'] == ball_count]) if not train_df.empty else 0
        val_count = len(val_df[val_df['ball_count'] == ball_count]) if not val_df.empty else 0
        
        if total_count > 0:
            train_ratio = train_count / total_count * 100
            val_ratio = val_count / total_count * 100
            print(f"Ball count {ball_count}: total={total_count}, train={train_count}({train_ratio:.1f}%), val={val_count}({val_ratio:.1f}%)")
        else:
            print(f"Ball count {ball_count}: total={total_count}, no samples")
    
    # ============= Feature 1: Generate stratified training subsets =============
    print(f"\n=== Feature 1: Generate stratified training subsets ===")
    
    # Define subset ratios
    subset_ratios = [0.8, 0.5, 0.2]
    
    for ratio in subset_ratios:
        print(f"\nGenerating {int(ratio*100)}% training subset...")
        subset_train_samples = []
        
        # Sample from each ball count proportionally
        for ball_count in sorted(train_df['ball_count'].unique()):
            ball_train_samples = train_df[train_df['ball_count'] == ball_count]
            total_count = len(ball_train_samples)
            
            if total_count == 0:
                continue
            elif total_count == 1:
                # Only one sample — always include it
                subset_train_samples.append(ball_train_samples)
                print(f"  Ball count {ball_count}: {total_count} -> {total_count} (only 1 sample)")
            else:
                # Sample at least 1; at most total_count
                target_count = max(1, int(total_count * ratio))
                
                if target_count >= total_count:
                    # Include all samples if target exceeds total
                    subset_train_samples.append(ball_train_samples)
                    print(f"  Ball count {ball_count}: {total_count} -> {total_count} (include all)")
                else:
                    # Random sample without replacement
                    subset_samples = ball_train_samples.sample(n=target_count, random_state=42)
                    subset_train_samples.append(subset_samples)
                    print(f"  Ball count {ball_count}: {total_count} -> {target_count} ({target_count/total_count*100:.1f}%)")
        
        # Concatenate and save subset
        if subset_train_samples:
            subset_train_df = pd.concat(subset_train_samples, ignore_index=True)
            
            subset_csv_path = Path(output_dir) / f"ball_counting_dataset_train_{int(ratio*100)}.csv"
            subset_train_df.to_csv(subset_csv_path, index=False, encoding='utf-8')
            print(f"Saved {int(ratio*100)}% subset to: {subset_csv_path} ({len(subset_train_df)} samples)")
            
            print(f"  {int(ratio*100)}% subset ball count distribution:")
            subset_counts = subset_train_df['ball_count'].value_counts().sort_index()
            print(f"  {subset_counts}")
        else:
            print(f"Warning: {int(ratio*100)}% subset is empty.")
    
    # ============= Feature 2: Generate single-sample-per-label validation sets =============
    print(f"\n=== Feature 2: Generate single-sample-per-label validation sets ===")
    
    # Generate 3 versions with different random seeds
    for version in range(1, 4):
        print(f"\nGenerating version {version} single-sample validation set...")
        single_val_samples = []
        
        # Pick one sample per ball count label
        for ball_count in range(1, 11):
            ball_val_samples = val_df[val_df['ball_count'] == ball_count]
            
            if len(ball_val_samples) == 0:
                print(f"  Ball count {ball_count}: no validation samples — skipping.")
                continue
            elif len(ball_val_samples) == 1:
                # Only one sample — use it directly
                selected_sample = ball_val_samples
                print(f"  Ball count {ball_count}: selected the only available sample.")
            else:
                # Random selection with a version-specific seed
                random_state = 42 + version * 10 + ball_count
                selected_sample = ball_val_samples.sample(n=1, random_state=random_state)
                print(f"  Ball count {ball_count}: randomly selected 1 from {len(ball_val_samples)} samples.")
            
            single_val_samples.append(selected_sample)
        
        # Concatenate and save
        if single_val_samples:
            single_val_df = pd.concat(single_val_samples, ignore_index=True)
            
            single_csv_path = Path(output_dir) / f"ball_counting_dataset_val_single_per_label_v{version}.csv"
            single_val_df.to_csv(single_csv_path, index=False, encoding='utf-8')
            print(f"Saved version {version} single-sample val set to: {single_csv_path} ({len(single_val_df)} samples)")
            
            print(f"  Version {version} ball count distribution:")
            single_counts = single_val_df['ball_count'].value_counts().sort_index()
            print(f"  {single_counts}")
        else:
            print(f"Warning: version {version} single-sample validation set is empty.")
    
    return df, train_df, val_df

def main():
    """
    Main entry point — example usage.
    """
    # Set the data root directory (update to your actual path)
    root_directory = "scratch/Ball_counting_CNN/ball_data_collection"
    
    # Set the output directory for generated CSV files
    output_directory = "scratch/Ball_counting_CNN/Tools_script"
    
    print("Starting ball-counting dataset collection...")
    print(f"Root directory: {root_directory}")
    print(f"Output directory: {output_directory}")
    
    # Validate root directory
    if not Path(root_directory).exists():
        print(f"Error: root directory {root_directory} not found.")
        print("Please update root_directory to your actual data path.")
        return
    
    # Collect data and generate CSVs
    try:
        all_data, train_data, val_data = collect_ball_samples(root_directory, output_directory)
        print("\nData collection complete!")
        
        # Show a preview of the collected data
        print(f"\nFirst 5 samples:")
        print(all_data.head())
        
        # Summarise generated files
        print(f"\n=== Generated files summary ===")
        output_path = Path(output_directory)
        generated_files = [
            "ball_counting_dataset_all.csv",
            "ball_counting_dataset_train.csv", 
            "ball_counting_dataset_val.csv",
            "ball_counting_dataset_train_80.csv",
            "ball_counting_dataset_train_50.csv",
            "ball_counting_dataset_train_20.csv",
            "ball_counting_dataset_val_single_per_label_v1.csv",
            "ball_counting_dataset_val_single_per_label_v2.csv",
            "ball_counting_dataset_val_single_per_label_v3.csv"
        ]
        
        print("Generated CSV files:")
        for filename in generated_files:
            file_path = output_path / filename
            if file_path.exists():
                print(f"✓ {filename}")
            else:
                print(f"✗ {filename} (not generated)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()