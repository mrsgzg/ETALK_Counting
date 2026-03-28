import os
import sys
import subprocess
from pathlib import Path

# Discovery settings
EPOCHS = [6,30,250]
INCLUDE_SUBSTR = "seed2048"  # discover experiments whose folder name contains this
## PREFIX_FILTER = "EC_"        # only experiments whose names start with this prefix
PREFIX_FILTER = "EC_50pct"
def discover_exp_dirs(experiments_root: Path, include_substr: str, prefix: str | None = None) -> list[str]:
    """Return absolute paths to experiment dirs under experiments_root whose names
    contain include_substr and which have a checkpoints/ subdir."""
    exp_dirs: list[str] = []
    if not experiments_root.exists():
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

# Quick knobs
DEVICE = "cpu"            # "auto" | "cpu" | "cuda"
CAM_TIMESTEP = "first"       # "last" | "all" | "first" | "mid"
SAMPLES_PER_CLASS = 0       # number of samples per predicted class for Grad-CAM


def derive_output_dir_from_ckpt(ckpt_path: str) -> str:
    # experiments/<exp_name>/checkpoints/epoch_XXX.pt -> Visualization/Embodied/<exp_name>
    p = Path(ckpt_path).resolve()
    exp_dir = p.parent.parent  # go up from checkpoints/<file>.pt to <exp_dir>
    exp_name = exp_dir.name
    base_vis = Path(__file__).resolve().parent / "Visualization" / "new_Embodied_multi_layer_JPCA"
    out = base_vis / exp_name
    return str(out)


def build_cmd(analyze_script: str, ckpt: str, out_dir: str) -> list:
    cmd = [
        sys.executable, analyze_script,
        "--checkpoint", ckpt,
        "--output_dir", out_dir,
        "--device", DEVICE,
        "--cam_timestep", CAM_TIMESTEP,
        "--gradcam_samples_per_class", str(SAMPLES_PER_CLASS),
    ]
    return cmd


def main():
    script_dir = Path(__file__).resolve().parent
    analyze_script = str((script_dir / "analyze_embodied_v2.py").resolve())
    experiments_root = script_dir / "experiments"

    # Auto-discover experiments containing the seed substring
    exp_dirs = discover_exp_dirs(experiments_root, INCLUDE_SUBSTR, prefix=PREFIX_FILTER)
    if not exp_dirs:
        print(f"No experiments found under {experiments_root} starting with '{PREFIX_FILTER}' and containing '{INCLUDE_SUBSTR}'.")
        return

    # Auto-build the checkpoint list from discovered EXP_DIRS × EPOCHS
    checkpoints = [
        str(Path(exp_dir) / "checkpoints" / f"epoch_{ep}.pt")
        for exp_dir in exp_dirs for ep in EPOCHS
    ]

    if not checkpoints:
        print("No checkpoints resolved from discovered experiments.")
        return

    print("Discovered experiments (seed filter applied):")
    for exp in exp_dirs:
        print(f"  - {exp}")

    for ckpt in checkpoints:
        if not Path(ckpt).exists():
            print(f"[skip] Missing checkpoint: {ckpt}")
            continue
        out_dir = derive_output_dir_from_ckpt(ckpt)
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "="*60)
        print(f"Analyzing: {ckpt}")
        print(f"Output dir: {out_dir}")
        print("="*60)
        cmd = build_cmd(analyze_script, ckpt, out_dir)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Analysis failed for {ckpt}: {e}")


if __name__ == "__main__":
    main()
