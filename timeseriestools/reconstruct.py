"""
GPS Timeseries Reconstruction Module
Uses SSA-based imputation and smoothing (integrates user's proven code).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try relative import first (for package), then absolute (for local use)
try:
    from .run_workflow import run_ssa_smoothing_workflow
    from .pca_imputation import *
except ImportError:
    from run_workflow import run_ssa_smoothing_workflow
    from pca_imputation import *


def detect_rate_outliers(cumulative_series, threshold=3.5, use_time=True):
    """
    Detect outliers using rate of change.
    
    Args:
        cumulative_series: pd.Series with DatetimeIndex
        threshold: MAD threshold (default 3.5)
        use_time: Use time-normalized rates
    
    Returns:
        pd.Series with outliers marked as NaN
    """
    original_nan_mask = cumulative_series.isna()
    valid_indices = np.where(~original_nan_mask)[0]
    
    if len(valid_indices) < 3:
        return cumulative_series
    
    rates = pd.Series(index=cumulative_series.index, dtype=float)
    
    for i in range(1, len(valid_indices)):
        curr_idx = valid_indices[i]
        prev_idx = valid_indices[i - 1]
        
        value_change = cumulative_series.iloc[curr_idx] - cumulative_series.iloc[prev_idx]
        
        if use_time and isinstance(cumulative_series.index, pd.DatetimeIndex):
            time_elapsed = (cumulative_series.index[curr_idx] - cumulative_series.index[prev_idx]).total_seconds() / (24 * 3600)
            if time_elapsed > 0:
                rates.iloc[curr_idx] = value_change / time_elapsed
        else:
            time_gap = curr_idx - prev_idx
            rates.iloc[curr_idx] = value_change / time_gap
    
    valid_rates = rates.dropna()
    if len(valid_rates) < 3:
        return cumulative_series
    
    median_rate = valid_rates.median()
    mad = np.median(np.abs(valid_rates - median_rate))
    
    if mad < 1e-10:
        return cumulative_series
    
    modified_z = 0.6745 * (rates - median_rate) / mad
    is_outlier = np.abs(modified_z) > threshold
    
    result = cumulative_series.copy()
    outlier_indices = np.where(is_outlier & ~original_nan_mask)[0]
    result.iloc[outlier_indices] = np.nan
    
    return result


def plot_reconstruction(original, outliers_removed, final, save_path):
    """Plot reconstruction steps."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    
    # Original
    axes[0].plot(original, 'o', ms=2, alpha=0.5, color='steelblue')
    axes[0].set_ylabel('Displacement (m)', fontsize=10, fontweight='bold')
    axes[0].set_title('Jump-Corrected Input', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Outliers removed
    axes[1].plot(outliers_removed, 'o', ms=2, alpha=0.5, color='darkorange')
    axes[1].set_ylabel('Displacement (m)', fontsize=10, fontweight='bold')
    axes[1].set_title('Outliers Removed', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Final SSA
    axes[2].plot(original, 'o', ms=1.5, alpha=0.1, color='lightgray', label='Original')
    axes[2].plot(final, '-', lw=2.5, color='darkgreen', label='SSA Smoothed')
    axes[2].set_ylabel('Displacement (m)', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].set_title('Final Cleaned Timeseries (SSA)', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_reconstruction(
    timeseries,
    station_name,
    output_dir='03_final_cleaned',
    # Outlier detection
    outlier_threshold=3.5,
    # SSA parameters
    fixed_n_components=None,
    variance_threshold=0.95,
    max_components=5,
    smooth_observed=True,
):
    """
    Run full reconstruction using SSA workflow.
    
    Args:
        timeseries: pd.Series (jump-corrected)
        station_name: Station ID
        output_dir: Output directory
        outlier_threshold: MAD threshold (2.0-5.0)
        fixed_n_components: Fixed SSA components (None for auto)
        variance_threshold: Variance threshold (0.8-0.95)
        max_components: Max SSA components (3-8)
        smooth_observed: Full smoothing (True) or gap-fill only (False)
    
    Returns:
        final: Fully cleaned timeseries
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION: {station_name}")
    print(f"{'='*60}")
    
    # Step 1: Remove outliers
    print("\n[1/2] Removing outliers...")
    outliers_removed = detect_rate_outliers(timeseries, threshold=outlier_threshold)
    n_outliers = (timeseries.notna() & outliers_removed.isna()).sum()
    print(f"  Removed {n_outliers} outliers")
    
    # Step 2: SSA smoothing & imputation
    print("\n[2/2] SSA smoothing & imputation...")
    try:
        results = run_ssa_smoothing_workflow(
            data=outliers_removed,
            fixed_n_components=fixed_n_components,
            variance_threshold=variance_threshold,
            max_components=max_components,
            smooth_observed=smooth_observed,
        )
        
        final = results['imputed_series']
        params = results['parameters']
        
        print(f"  SSA parameters: L={params['embedding_dim']}, r={params['n_components']}")
        
    except Exception as e:
        print(f"  SSA failed: {e}")
        print("  Falling back to simple interpolation")
        final = outliers_removed.interpolate(method='linear')
    
    # Metrics
    valid_orig = timeseries.dropna()
    valid_final = final.loc[valid_orig.index]
    rmse = np.sqrt(np.mean((valid_orig - valid_final)**2))
    
    print(f"\nQuality:")
    print(f"  RMSE: {rmse*1000:.2f} mm")
    
    # Save
    csv_path = output_path / f"{station_name}_final.csv"
    final.to_csv(csv_path, header=True)
    
    plot_path = output_path / f"{station_name}_reconstruction.png"
    plot_reconstruction(timeseries, outliers_removed, final, plot_path)
    
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {plot_path}")
    
    return final
