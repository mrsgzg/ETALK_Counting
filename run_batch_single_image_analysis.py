"""
批量单图像分析脚本
功能: 自动发现所有实验，提取各epoch的checkpoint进行单图像模型分析
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置参数 ====================
# 实验发现设置
EPOCHS = [5, 29, 249]                    # 要分析的epoch列表
INCLUDE_SUBSTR = "seed2048"              # 发现包含此substring的实验
PREFIX_FILTER = "SI_"                    # 只发现以此prefix开头的实验

# 快速可调参数
DEVICE = "cpu"                           # "auto" | "cpu" | "cuda"
BATCH_SIZE = 32                          # 批次大小
NUM_WORKERS = 4                          # 数据加载器工作进程数
GRADCAM_SAMPLES_PER_CLASS = 2           # 每个类别的Grad-CAM样本数
ERROR_SAMPLES = 20                       # 错误样本可视化数量
N_SAMPLES = -1                           # 分析样本数，-1表示全部


# ==================== 发现实验函数 ====================
def discover_exp_dirs(experiments_root: Path, include_substr: str, prefix: str | None = None) -> list[str]:
    """
    发现实验目录
    
    Args:
        experiments_root: 实验根目录
        include_substr: 实验名称必须包含的子串
        prefix: 实验名称必须以此开头（可选）
    
    Returns:
        实验目录的绝对路径列表
    """
    exp_dirs: list[str] = []
    if not experiments_root.exists():
        print(f"警告: 实验根目录不存在: {experiments_root}")
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
    从checkpoint路径推导输出目录
    
    Args:
        ckpt_path: checkpoint文件路径
               experiments/<exp_name>/checkpoints/epoch_XXX.pt
    
    Returns:
        输出目录路径
               Visualization/Single_image_multi_layer/<exp_name>_epoch_XXX/
    """
    p = Path(ckpt_path).resolve()
    exp_dir = p.parent.parent  # 从 checkpoints/<file>.pt 向上至 <exp_dir>
    exp_name = exp_dir.name
    
    # 从checkpoint文件名提取epoch信息 (epoch_XXX.pt)
    ckpt_filename = p.stem  # 去掉.pt后缀，e.g., "epoch_6", "epoch_30"
    
    # 使用 Visualization/Single_image_multi_layer 目录，目录名包含实验名和epoch
    base_vis = Path(__file__).resolve().parent / "Visualization" / "Single_image_multi_layer"
    out = base_vis / f"{exp_name}_{ckpt_filename}"
    return str(out)


def build_cmd(analyze_script: str, ckpt: str, out_dir: str) -> list:
    """
    构建分析命令
    
    Args:
        analyze_script: analyze_single_image.py脚本路径
        ckpt: checkpoint文件路径
        out_dir: 输出目录
    
    Returns:
        命令列表
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
    """主函数"""
    script_dir = Path(__file__).resolve().parent
    analyze_script = str((script_dir / "analyze_single_image.py").resolve())
    experiments_root = script_dir / "experiments"
    
    print("\n" + "="*70)
    print("批量单图像分析脚本")
    print("="*70)
    print(f"脚本位置: {__file__}")
    print(f"分析脚本: {analyze_script}")
    print(f"实验根目录: {experiments_root}")
    print(f"输出目录: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print("="*70 + "\n")
    
    # 参数配置显示
    print("分析配置:")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  INCLUDE_SUBSTR: {INCLUDE_SUBSTR}")
    print(f"  PREFIX_FILTER: {PREFIX_FILTER}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  GRADCAM_SAMPLES_PER_CLASS: {GRADCAM_SAMPLES_PER_CLASS}")
    print(f"  ERROR_SAMPLES: {ERROR_SAMPLES}")
    print(f"  N_SAMPLES: {N_SAMPLES if N_SAMPLES > 0 else '全部'}")
    print()
    
    # 验证分析脚本存在
    if not Path(analyze_script).exists():
        print(f"错误: 分析脚本不存在: {analyze_script}")
        return
    
    # 自动发现包含seed substring的实验
    exp_dirs = discover_exp_dirs(experiments_root, INCLUDE_SUBSTR, prefix=PREFIX_FILTER)
    
    if not exp_dirs:
        print(f"警告: 未发现实验")
        print(f"  位置: {experiments_root}")
        print(f"  过滤条件: 前缀={PREFIX_FILTER}, 包含={INCLUDE_SUBSTR}")
        return
    
    # 自动构建checkpoint列表
    checkpoints = [
        str(Path(exp_dir) / "checkpoints" / f"epoch_{ep}.pt")
        for exp_dir in exp_dirs for ep in EPOCHS
    ]
    
    if not checkpoints:
        print("错误: 未解析到任何checkpoint")
        return
    
    print(f"发现的实验 (应用seed过滤后):")
    for exp in exp_dirs:
        print(f"  - {exp}")
    print()
    
    print(f"要分析的checkpoints (共 {len(checkpoints)} 个):")
    for ckpt in sorted(checkpoints):
        if Path(ckpt).exists():
            print(f"  ✓ {ckpt}")
        else:
            print(f"  ✗ [缺失] {ckpt}")
    print()
    
    # 统计信息
    existing_count = sum(1 for ckpt in checkpoints if Path(ckpt).exists())
    missing_count = len(checkpoints) - existing_count
    
    if missing_count > 0:
        print(f"警告: {missing_count}/{len(checkpoints)} 个checkpoint缺失")
    
    print(f"开始分析... (存在的checkpoints: {existing_count})\n")
    
    # 逐个分析checkpoint
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, ckpt in enumerate(sorted(checkpoints), 1):
        ckpt_path = Path(ckpt)
        
        if not ckpt_path.exists():
            print(f"\n[{idx}/{len(checkpoints)}] [跳过] Checkpoint不存在: {ckpt}")
            skipped += 1
            continue
        
        out_dir = derive_output_dir_from_ckpt(ckpt)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"[{idx}/{len(checkpoints)}] 分析中...")
        print("="*70)
        print(f"Checkpoint: {ckpt}")
        print(f"输出目录: {out_dir}")
        print("="*70)
        
        cmd = build_cmd(analyze_script, ckpt, out_dir)
        
        try:
            # 运行分析
            result = subprocess.run(cmd, check=True)
            successful += 1
            print(f"✓ 分析成功")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"✗ 分析失败 (返回码: {e.returncode})")
            print(f"  错误信息: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ 分析异常: {e}")
    
    # 最终统计
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print(f"统计信息:")
    print(f"  成功: {successful}")
    print(f"  失败: {failed}")
    print(f"  跳过: {skipped}")
    print(f"  总计: {len(checkpoints)}")
    print(f"\n所有结果已保存到: {script_dir / 'Visualization' / 'Single_image_multi_layer'}")
    print()



if __name__ == "__main__":
    main()
