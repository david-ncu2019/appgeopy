"""
GPS Jump Correction Module
Corrects timeseries based on manually selected jump dates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def correct_jumps(timeseries, jump_dates):
    """
    Correct jumps using robust median alignment.
    
    Args:
        timeseries: pd.Series with DatetimeIndex
        jump_dates: List of jump date strings (e.g., ['2022-06-17'])
    
    Returns:
        corrected: Jump-corrected timeseries
        offsets: List of offsets applied (in meters)
    """
    corrected = timeseries.copy()
    
    if not jump_dates:
        return corrected, []
    
    # Convert dates to indices
    jump_dates_dt = [pd.to_datetime(d) for d in jump_dates]
    jump_indices = [timeseries.index.get_loc(date) for date in jump_dates_dt 
                   if date in timeseries.index]
    
    if not jump_indices:
        return corrected, []
    
    jump_indices = sorted(jump_indices)
    
    # Create segments
    segments = []
    start = 0
    for jump_idx in jump_indices:
        segments.append((start, jump_idx))
        start = jump_idx
    segments.append((start, len(timeseries)))
    
    # Calculate and apply offsets
    offsets = [0.0]
    values = timeseries.values
    
    for i in range(1, len(segments)):
        seg_start, seg_end = segments[i]
        prev_start, prev_end = segments[i-1]
        
        curr_vals = values[seg_start:seg_end]
        prev_vals = values[prev_start:prev_end]
        
        curr_valid = curr_vals[~np.isnan(curr_vals)]
        prev_valid = prev_vals[~np.isnan(prev_vals)]
        
        if len(curr_valid) < 10 or len(prev_valid) < 10:
            offsets.append(0.0)
            continue
        
        n_edge = min(30, len(prev_valid)//2, len(curr_valid)//2)
        
        prev_edge = np.median(prev_valid[-n_edge:])
        curr_edge = np.median(curr_valid[:n_edge])
        offset = prev_edge - curr_edge
        
        offsets.append(offset)
        corrected.iloc[seg_start:seg_end] += offset
    
    return corrected, offsets


def plot_jump_correction(original, corrected, jump_dates, offsets, save_path):
    """Plot original vs corrected."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    jump_dates_dt = [pd.to_datetime(d) for d in jump_dates]
    
    # Original
    axes[0].plot(original, 'o', ms=2, alpha=0.5, color='steelblue')
    for jd in jump_dates_dt:
        axes[0].axvline(jd, color='red', ls='--', lw=2.5, alpha=0.8)
    axes[0].set_ylabel('Displacement (m)', fontsize=11, fontweight='bold')
    axes[0].set_title('Original Data + Selected Jumps', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Corrected
    axes[1].plot(corrected, 'o', ms=2, alpha=0.5, color='darkgreen')
    axes[1].set_ylabel('Displacement (m)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[1].set_title('Jump-Corrected Data', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_jump_correction(timeseries, station_name, jump_dates, output_dir='02_jump_corrected', savefig=False):
    """
    Correct jumps and save results.
    
    Returns:
        corrected: Jump-corrected timeseries
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"JUMP CORRECTION: {station_name}")
    print(f"{'='*60}")
    
    # Correct jumps
    corrected, offsets = correct_jumps(timeseries, jump_dates)
    
    # Print results
    print(f"Jumps: {len(jump_dates)}")
    for i, (date, offset) in enumerate(zip(jump_dates, offsets[1:]), 1):
        print(f"  {i}. {date}: {offset*1000:.1f} mm")
    
    # Save
    csv_path = output_path / f"{station_name}_corrected.csv"
    corrected.to_csv(csv_path, header=True)
    
    if savefig:
        plot_path = output_path / f"{station_name}_correction.png"
        plot_jump_correction(timeseries, corrected, jump_dates, offsets, plot_path)
    
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    if savefig:
        print(f"  {plot_path}")
    
    return corrected
