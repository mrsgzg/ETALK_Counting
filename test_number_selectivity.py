#!/usr/bin/env python3
"""
Quick test script for number selectivity and RSA analysis
"""
import numpy as np
import tempfile
import os
import sys

CUR_DIR = os.path.dirname(__file__)
sys.path.append(CUR_DIR)

from analyze_embodied import analyze_number_selectivity, compute_number_line_rsa, analyze_rotational_dynamics, perform_jPCA_analysis
from sklearn.decomposition import PCA

# Create synthetic test data
print("Creating synthetic test data...")
N_samples = 100  # samples
T_timesteps = 11  # sequence length
H_hidden = 512   # hidden units
num_classes = 11  # ball counts 0-10

# Generate synthetic LSTM hidden states
# Make some units number-selective
np.random.seed(42)
hidden_btH = np.random.randn(N_samples, T_timesteps, H_hidden) * 0.1

# Make first 20 units selective for different numbers
# AND make representations ordered (for number line effect)
for unit_idx in range(20):
    pref_count = unit_idx % 11
    for sample_idx in range(N_samples):
        label = sample_idx % 11
        if label == pref_count:
            hidden_btH[sample_idx, -1, unit_idx] += 2.0
        else:
            hidden_btH[sample_idx, -1, unit_idx] -= 0.5

# Add ordered structure for number line (units 20-50)
for unit_idx in range(20, 51):
    for sample_idx in range(N_samples):
        label = sample_idx % 11
        # Activation increases with count (linear ordering)
        hidden_btH[sample_idx, -1, unit_idx] = label * 0.5 + np.random.randn() * 0.1

# Generate labels
labels = np.array([i % 11 for i in range(N_samples)])

print(f"hidden_btH shape: {hidden_btH.shape}")
print(f"labels shape: {labels.shape}")
print(f"Label distribution: {np.bincount(labels)}")

print("\n" + "="*60)
print("Test 1: Number Selectivity Analysis")
print("="*60)

with tempfile.TemporaryDirectory() as tmpdir:
    result1 = analyze_number_selectivity(
        hidden_btH=hidden_btH,
        labels=labels,
        save_dir=tmpdir,
        layer_name="TestLayer",
        top_k=50
    )
    
    print(f"\nResults:")
    print(f"  tuning_curves shape: {result1['tuning_curves'].shape}")
    print(f"  selectivity shape: {result1['selectivity'].shape}")
    print(f"  top_units: {result1['top_units'][:5]}...")
    print(f"  percent_selective: {result1['percent_selective']:.1f}%")

print("\n" + "="*60)
print("Test 2: Number Line RSA Analysis")
print("="*60)

with tempfile.TemporaryDirectory() as tmpdir:
    result2 = compute_number_line_rsa(
        hidden_btH=hidden_btH,
        labels=labels,
        save_dir=tmpdir,
        layer_name="TestLayer"
    )
    
    print(f"\nResults:")
    print(f"  rdm shape: {result2['rdm'].shape}")
    print(f"  correlation (ρ): {result2['correlation']:.4f}")
    print(f"  p_value: {result2['p_value']:.6f}")
    print(f"  mean_representations shape: {result2['mean_representations'].shape}")
    
    # Check output files
    print(f"\nOutput files:")
    for f in sorted(os.listdir(tmpdir)):
        size = os.path.getsize(os.path.join(tmpdir, f))
        print(f"  {f}: {size} bytes")

print("\n" + "="*60)
print("Test 3: Rotational Dynamics Analysis")
print("="*60)

# Create synthetic data with circular trajectories (strong rotation)
print("Creating synthetic circular trajectory data...")
np.random.seed(42)
hidden_circular = np.random.randn(N_samples, T_timesteps, H_hidden) * 0.1

# Create circular trajectories in first 2 dimensions
for sample_idx in range(N_samples):
    label = sample_idx % 11
    radius = 2.0 + label * 0.5  # Different radius per count
    for t in range(T_timesteps):
        angle = (t / T_timesteps) * 2 * np.pi  # Full rotation
        hidden_circular[sample_idx, t, 0] = radius * np.cos(angle)
        hidden_circular[sample_idx, t, 1] = radius * np.sin(angle)

# Fit PCA on final timestep (just for interface compatibility)
final_states = hidden_circular[:, -1, :].reshape(N_samples, H_hidden)
pca2d = PCA(n_components=2)
pca2d.fit(final_states)

print(f"hidden_circular shape: {hidden_circular.shape}")

with tempfile.TemporaryDirectory() as tmpdir:
    result3 = analyze_rotational_dynamics(
        hidden_btH=hidden_circular,
        labels=labels,
        pca2d_model=pca2d,
        save_dir=tmpdir,
        layer_name="TestLayer"
    )
    
    print(f"\nResults:")
    print(f"  mean_trajectories_2d shape: {result3['mean_trajectories_2d'].shape}")
    print(f"  velocities_2d shape: {result3['velocities_2d'].shape}")
    print(f"  angles: {len(result3['angles'])} angles computed")
    print(f"  rotation_score: {result3['rotation_score']:.4f} (1=strong rotation)")
    print(f"  mean_angle: {result3['mean_angle']:.2f}° (90°=perfect rotation)")
    
    # Check output files
    print(f"\nOutput files:")
    for f in sorted(os.listdir(tmpdir)):
        size = os.path.getsize(os.path.join(tmpdir, f))
        print(f"  {f}: {size} bytes")

print("\n" + "="*60)
print("Test 4: jPCA Analysis (Full Churchland Method)")
print("="*60)

# Use the same circular trajectory data as Test 3
print("Using circular trajectory data for jPCA test...")

with tempfile.TemporaryDirectory() as tmpdir:
    result4 = perform_jPCA_analysis(
        hidden_btH=hidden_circular,
        labels=labels,
        save_dir=tmpdir,
        layer_name="TestLayer",
        pca_n_components=6
    )
    
    print(f"\nResults:")
    print(f"  jPCs shape: {result4['jPCs'].shape}")
    print(f"  M_skew shape: {result4['M_skew'].shape}")
    print(f"  eigenvalues shape: {result4['eigenvalues'].shape}")
    print(f"  rotation_frequency_omega: {result4['rotation_frequency_omega']:.4f} rad/step")
    print(f"  rotation_variance_fraction: {result4['rotation_variance_fraction']:.1%}")
    print(f"  dynamics_fit_R2: {result4['dynamics_fit_R2']:.4f}")
    print(f"  rotation_quality: {result4['rotation_quality']:.4f}")
    print(f"  trajectories_jPCA: {len(result4['trajectories_jPCA'])} conditions")
    
    # Check output files
    print(f"\nOutput files:")
    for f in sorted(os.listdir(tmpdir)):
        size = os.path.getsize(os.path.join(tmpdir, f))
        print(f"  {f}: {size} bytes")

print("\n✓ All tests completed successfully!")
