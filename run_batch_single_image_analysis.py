"""
info
info: infoïepochcheckpoint
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== info ====================
# info
EPOCHS = [5, 29, 249]                    # epoch
INCLUDE_SUBSTR = "seed2048"              # substring
PREFIX_FILTER = "SI_"                    # prefix

# info
DEVICE = "cpu"                           # "auto" | "cpu" | "cuda"
BATCH_SIZE = 32                          # info
NUM_WORKERS = 4                          # info
GRADCAM_SAMPLES_PER_CLASS = 2           # Grad-CAM
ERROR_SAMPLES = 20                       # info
N_SAMPLES = -1                           # infoïinfo-1


# ==================== info ====================
def discover_exp_dirs(experiments_root: Path, include_substr: str, prefix: str | None = None) -> list[str]:
    """
    info
    
    Args:
        experiments_root: info
        include_substr: info
        prefix: infoïinfoïinfo
    
    Returns:
        info
    """
    exp_dirs: list[str] = []
    if not experiments_root.exists():
        print(f"info: info: {experiments_root}")
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
    checkpoint
    
    Args:
        ckpt_path: checkpoint
               experiments/<exp_name>/checkpoints/epoch_XXX.pt
    
    Returns:
        info
               Visualization/Single_image_multi_layer/<exp_name>_epoch_XXX/
    """
    p = Path(ckpt_path).resolve()
    exp_dir = p.parent.parent  # info checkpoints/<file>.pt info <exp_dir>
    exp_name = exp_dir.name
    
    # checkpointepoch (epoch_XXX.pt)
    ckpt_filename = p.stem  # info.ptïe.g., "epoch_6", "epoch_30"
    
    # info Visualization/Single_image_multi_layer infoïepoch
    base_vis = Path(__file__).resolve().parent / "Visualization" / "Single_image_multi_layer"
    out = base_vis / f"{exp_name}_{ckpt_filename}"
    return str(out)


def build_cmd(analyze_script: str, ckpt: str, out_dir: str) -> list:
    """
    info
    
    Args:
        analyze_script: analyze_single_image.py
        ckpt: checkpoint
        out_dir: info
    
    Returns:
        info
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
    """info"""
    script_dir = Path(__file__).resolve().parent
    analyze_script = str((script_dir / "analyze_single_image.py").resolve())
    experiments_root = script_dir / "experiments"
    
    print("\n" + "="*70)
    print("info")
    print("="*70)
    print(f"info: {__file__}")
    print(f"info: {analyze_script}")
    print(f"info: {experiments_root}")
    print(f"info: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print("="*70 + "\n")
    
    # info
    print("info:")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  INCLUDE_SUBSTR: {INCLUDE_SUBSTR}")
    print(f"  PREFIX_FILTER: {PREFIX_FILTER}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  GRADCAM_SAMPLES_PER_CLASS: {GRADCAM_SAMPLES_PER_CLASS}")
    print(f"  ERROR_SAMPLES: {ERROR_SAMPLES}")
    print(f"  N_SAMPLES: {N_SAMPLES if N_SAMPLES > 0 else 'info'}")
    print()
    
    # info
    if not Path(analyze_script).exists():
        print(f"info: info: {analyze_script}")
        return
    
    # seed substring
    exp_dirs = discover_exp_dirs(experiments_root, INCLUDE_SUBSTR, prefix=PREFIX_FILTER)
    
    if not exp_dirs:
        print(f"info: info")
        print(f"  info: {experiments_root}")
        print(f"  info: info={PREFIX_FILTER}, info={INCLUDE_SUBSTR}")
        return
    
    # checkpoint
    checkpoints = [
        str(Path(exp_dir) / "checkpoints" / f"epoch_{ep}.pt")
        for exp_dir in exp_dirs for ep in EPOCHS
    ]
    
    if not checkpoints:
        print("info: checkpoint")
        return
    
    print(f"info (seed):")
    for exp in exp_dirs:
        print(f"  - {exp}")
    print()
    
    print(f"checkpoints (info {len(checkpoints)} info):")
    for ckpt in sorted(checkpoints):
        if Path(ckpt).exists():
            print(f"  info {ckpt}")
        else:
            print(f"  info [info] {ckpt}")
    print()
    
    # info
    existing_count = sum(1 for ckpt in checkpoints if Path(ckpt).exists())
    missing_count = len(checkpoints) - existing_count
    
    if missing_count > 0:
        print(f"info: {missing_count}/{len(checkpoints)} checkpoint")
    
    print(f"info... (checkpoints: {existing_count})\n")
    
    # checkpoint
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, ckpt in enumerate(sorted(checkpoints), 1):
        ckpt_path = Path(ckpt)
        
        if not ckpt_path.exists():
            print(f"\n[{idx}/{len(checkpoints)}] [info] Checkpoint: {ckpt}")
            skipped += 1
            continue
        
        out_dir = derive_output_dir_from_ckpt(ckpt)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"[{idx}/{len(checkpoints)}] info...")
        print("="*70)
        print(f"Checkpoint: {ckpt}")
        print(f"info: {out_dir}")
        print("="*70)
        
        cmd = build_cmd(analyze_script, ckpt, out_dir)
        
        try:
            # info
            result = subprocess.run(cmd, check=True)
            successful += 1
            print(f"info info")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"info info (info: {e.returncode})")
            print(f"  info: {e}")
        except Exception as e:
            failed += 1
            print(f"info info: {e}")
    
    # info
    print("\n" + "="*70)
    print("infoïinfo")
    print("="*70)
    print(f"info:")
    print(f"  info: {successful}")
    print(f"  info: {failed}")
    print(f"  info: {skipped}")
    print(f"  info: {len(checkpoints)}")
    print(f"\n: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print()



if __name__ == "__main__":
    main()
