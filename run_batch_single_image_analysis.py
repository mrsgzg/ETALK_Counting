"""
Batch analysis runner for single-image classifier experiments.
Discovers experiment directories and runs the analysis script for each checkpoint.
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== Configuration ====================
# Experiment discovery settings
EPOCHS = [5, 29, 249]                    # epoch indices to analyse
INCLUDE_SUBSTR = "seed2048"              # only include experiments whose folder name contains this
PREFIX_FILTER = "SI_"                    # only include experiments whose folder name starts with this

# Analysis settings
DEVICE = "cpu"                           # "auto" | "cpu" | "cuda"
BATCH_SIZE = 32                          # inference batch size
NUM_WORKERS = 4                          # DataLoader worker processes
GRADCAM_SAMPLES_PER_CLASS = 2           # Grad-CAM samples per class
ERROR_SAMPLES = 20                       # number of error samples to visualise
N_SAMPLES = -1                           # subset size for Grad-CAM/errors; -1 uses all samples


# ==================== Helper functions ====================
def discover_exp_dirs(experiments_root: Path, include_substr: str, prefix: str | None = None) -> list[str]:
    """
    Discover valid experiment directories under experiments_root.
    
    Args:
        experiments_root: root directory containing experiment subdirectories
        include_substr: only include directories whose name contains this substring
        prefix: if provided, only include directories whose name starts with this prefix
    
    Returns:
        sorted list of absolute paths to matching experiment directories
    """
    exp_dirs: list[str] = []
    if not experiments_root.exists():
        print(f"Warning: experiments root not found: {experiments_root}")
        return exp_dirs
    
    for child in sorted(experiments_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if prefix and not name.startswith(prefix):
            continue
        if include_substr in name and (child / "checkpoints").is_dir():
            exp_dirs.append(str(child.resolve()))
    
    return exp_dirs


def derive_output_dir_from_ckpt(ckpt_path: str) -> str:
    """
    Derive the output directory for a given checkpoint path.
    
    Args:
        ckpt_path: path to the checkpoint file, expected structure:
               experiments/<exp_name>/checkpoints/epoch_XXX.pt
    
    Returns:
        output directory path:
               Visualization/Single_image_multi_layer/<exp_name>_epoch_XXX/
    """
    p = Path(ckpt_path).resolve()
    exp_dir = p.parent.parent  # navigate up: checkpoints/<file>.pt -> <exp_dir>
    exp_name = exp_dir.name
    
    # Extract epoch suffix from filename (e.g. "epoch_6", "epoch_30")
    ckpt_filename = p.stem  # filename without .pt extension
    
    # Place outputs under Visualization/Single_image_multi_layer, tagged with epoch
    base_vis = Path(__file__).resolve().parent / "Visualization" / "Single_image_multi_layer"
    out = base_vis / f"{exp_name}_{ckpt_filename}"
    return str(out)


def build_cmd(analyze_script: str, ckpt: str, out_dir: str) -> list:
    """
    Build the command list to invoke analyze_single_image.py.
    
    Args:
        analyze_script: path to analyze_single_image.py
        ckpt: path to the checkpoint file
        out_dir: output directory for this run
    
    Returns:
        list of command arguments suitable for subprocess.run
    """
    cmd = [
        sys.executable, analyze_script,
        "--checkpoint", ckpt,
        "--output_dir", out_dir,
        "--device", DEVICE,
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--n_samples", str(N_SAMPLES),
        "--gradcam_samples_per_class", str(GRADCAM_SAMPLES_PER_CLASS),
        "--error_samples", str(ERROR_SAMPLES),
    ]
    return cmd


def main():
    """Discover experiments and run analysis for each checkpoint."""
    script_dir = Path(__file__).resolve().parent
    analyze_script = str((script_dir / "analyze_single_image.py").resolve())
    experiments_root = script_dir / "experiments"
    
    print("\n" + "="*70)
    print("Single-Image Batch Analysis Runner")
    print("="*70)
    print(f"Script: {__file__}")
    print(f"Analysis script: {analyze_script}")
    print(f"Experiments root: {experiments_root}")
    print(f"Output root: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print("="*70 + "\n")
    
    # Print active configuration
    print("Configuration:")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  INCLUDE_SUBSTR: {INCLUDE_SUBSTR}")
    print(f"  PREFIX_FILTER: {PREFIX_FILTER}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  GRADCAM_SAMPLES_PER_CLASS: {GRADCAM_SAMPLES_PER_CLASS}")
    print(f"  ERROR_SAMPLES: {ERROR_SAMPLES}")
    print(f"  N_SAMPLES: {N_SAMPLES if N_SAMPLES > 0 else 'all samples'}")
    print()
    
    # Validate that the analysis script exists
    if not Path(analyze_script).exists():
        print(f"Error: analysis script not found: {analyze_script}")
        return
    
    # Discover experiment directories matching the seed substring and prefix
    exp_dirs = discover_exp_dirs(experiments_root, INCLUDE_SUBSTR, prefix=PREFIX_FILTER)
    
    if not exp_dirs:
        print(f"No matching experiments found.")
        print(f"  Experiments root: {experiments_root}")
        print(f"  Filters: prefix={PREFIX_FILTER}, substring={INCLUDE_SUBSTR}")
        return
    
    # Build the list of checkpoint paths to analyse
    checkpoints = [
        str(Path(exp_dir) / "checkpoints" / f"epoch_{ep}.pt")
        for exp_dir in exp_dirs for ep in EPOCHS
    ]
    
    if not checkpoints:
        print("No checkpoints resolved from discovered experiments.")
        return
    
    print(f"Discovered experiments (filtered by seed substring):")
    for exp in exp_dirs:
        print(f"  - {exp}")
    print()
    
    print(f"Checkpoints to process ({len(checkpoints)} total):")
    for ckpt in sorted(checkpoints):
        if Path(ckpt).exists():
            print(f"  [found]   {ckpt}")
        else:
            print(f"  [missing] {ckpt}")
    print()
    
    # Summarise availability
    existing_count = sum(1 for ckpt in checkpoints if Path(ckpt).exists())
    missing_count = len(checkpoints) - existing_count
    
    if missing_count > 0:
        print(f"Warning: {missing_count}/{len(checkpoints)} checkpoints not found and will be skipped.")
    
    print(f"Starting analysis ({existing_count} checkpoints to process)...\n")
    
    # Process each checkpoint
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, ckpt in enumerate(sorted(checkpoints), 1):
        ckpt_path = Path(ckpt)
        
        if not ckpt_path.exists():
            print(f"\n[{idx}/{len(checkpoints)}] [skipped] Checkpoint not found: {ckpt}")
            skipped += 1
            continue
        
        out_dir = derive_output_dir_from_ckpt(ckpt)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"[{idx}/{len(checkpoints)}] Running analysis...")
        print("="*70)
        print(f"Checkpoint: {ckpt}")
        print(f"Output dir: {out_dir}")
        print("="*70)
        
        cmd = build_cmd(analyze_script, ckpt, out_dir)
        
        try:
            result = subprocess.run(cmd, check=True)
            successful += 1
            print(f"Analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"Analysis failed (return code: {e.returncode})")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            print(f"Unexpected error: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("Batch analysis complete.")
    print("="*70)
    print(f"Results:")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    print(f"  Total:      {len(checkpoints)}")
    print(f"\nOutputs saved to: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print()


if __name__ == "__main__":
    main()