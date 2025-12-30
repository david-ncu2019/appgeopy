"""
GPS Jump Correction Module
Corrects time series based on manually selected jump dates.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def correct_jumps(
    timeseries: pd.Series,
    jump_dates: List[str]
) -> Tuple[pd.Series, List[float]]:
    """
    Correct jumps using robust median alignment.
    
    Args:
        timeseries: Time series with DatetimeIndex
        jump_dates: List of jump date strings (e.g., ['2022-06-17'])
    
    Returns:
        corrected: Jump-corrected time series
        offsets: List of offsets applied (in meters)
    """
    corrected = timeseries.copy()
    
    if not jump_dates:
        return corrected, []
    
    # Convert dates to indices
    jump_dates_dt = [pd.to_datetime(d) for d in jump_dates]
    jump_indices = [
        timeseries.index.get_loc(date) for date in jump_dates_dt 
        if date in timeseries.index
    ]
    
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


def plot_jump_correction(
    original: pd.Series,
    corrected: pd.Series,
    jump_dates: List[str],
    offsets: List[float],
    save_path: Path
) -> None:
    """
    Publication-quality plot comparing original and jump-corrected time series,
    annotated with correction magnitudes.
    """
    # 1. Global Publication Styling (serif fonts, high DPI)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300
    })

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0.1})
    
    # Convert string dates to datetime objects
    jump_dates_dt = [pd.to_datetime(d) for d in jump_dates]
    
    # Define professional color palette
    color_orig = '#34495e'  # Dark slate blue/gray
    color_corr = '#008080'  # Teal/Dark Cyan
    color_jump = '#c0392b'  # Deep red
    
    # --- Subplot 1: Original Data & Jump Events ---
    ax0 = axes[0]
    # Use slightly smaller markers and lower alpha for dense data
    ax0.plot(original.index, original.values, 'o', 
             ms=2.5, alpha=0.8, color=color_orig, markeredgewidth=0, label='Raw Data')
    
    # Iterate through dates and offsets simultaneously to annotate
    # We use enumerate to alternate label heights to avoid crowding
    for i, (jd, offset) in enumerate(zip(jump_dates_dt, offsets)):
        # Vertical line for the jump
        ax0.axvline(jd, color=color_jump, ls='-', lw=1.5, alpha=0.8)
        
        # Annotation showing the offset magnitude
        # Alternate vertical position to prevent overlap
        y_pos_factor = 1.05 if i % 2 == 0 else 1.12
        
        label_text = f"{offset:+.3f} m" # Format with +/- sign and 3 decimals
        
        ax0.text(
            x=jd, y=y_pos_factor, s=label_text,
            transform=ax0.get_xaxis_transform(), 
            color=color_jump, fontsize=9, fontweight='bold',
            ha='center', va='bottom', rotation=0,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_jump, alpha=0.9)
        )

    ax0.set_ylabel('Displacement (m)', fontweight='bold')
    # Use a descriptive title, move legend to a clean spot
    ax0.set_title(r'$\bf{a.}$ Original Time Series with Detected Jumps', loc='left')
    
    # Custom legend for top plot
    legend_elements_0 = [
        Line2D([0], [0], color=color_orig, marker='o', ls='None', alpha=0.6, label='Raw GPS Data'),
        Line2D([0], [0], color=color_jump, ls='-', lw=1.5, label='Detected Jump Event'),
    ]
    ax0.legend(handles=legend_elements_0, loc='upper left', frameon=True, fontsize=10)

    # --- Subplot 2: Corrected Data ---
    ax1 = axes[1]
    ax1.plot(corrected.index, corrected.values, 'o', 
             ms=2.5, alpha=0.8, color=color_corr, markeredgewidth=0, label='Corrected Data')
             
    ax1.set_ylabel('Displacement (m)', fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_title(r'$\bf{b.}$ Jump-Corrected Time Series', loc='left')
    
    # Simple legend for bottom plot
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # --- Common Styling Refinements ---
    for ax in axes:
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        # Remove top and right spines for a cleaner scientific look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Ensure y-axis limits match so visual comparison is accurate
        combined_min = min(original.min(), corrected.min())
        combined_max = max(original.max(), corrected.max())
        # Add a small buffer (5%)
        y_range = combined_max - combined_min
        ax.set_ylim(combined_min - 0.05*y_range, combined_max + 0.05*y_range)

    # Adjust layout to accommodate top labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save high-resolution output
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_jump_correction(
    timeseries: pd.Series,
    station_name: str,
    jump_dates: List[str],
    output_dir: str = '02_jump_corrected',
    savefig: bool = False,
    verbose: bool = True
) -> pd.Series:
    """
    Correct jumps and save results.
    
    Args:
        timeseries: Time series with DatetimeIndex
        station_name: Station identifier
        jump_dates: List of jump dates to correct
        output_dir: Output directory
        savefig: Save diagnostic plot
        verbose: Print progress messages
    
    Returns:
        corrected: Jump-corrected time series
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"JUMP CORRECTION: {station_name}")
        print(f"{'='*60}")
    
    # Correct jumps
    corrected, offsets = correct_jumps(timeseries, jump_dates)
    
    if verbose:
        print(f"Jumps: {len(jump_dates)}")
        for i, (date, offset) in enumerate(zip(jump_dates, offsets[1:]), 1):
            print(f"  {i}. {date}: {offset*1000:.1f} mm")
    
    # Save
    csv_path = output_path / f"{station_name}_corrected.csv"
    corrected.to_csv(csv_path, header=True)
    
    if savefig:
        plot_path = output_path / f"{station_name}_correction.png"
        plot_jump_correction(timeseries, corrected, jump_dates, offsets, plot_path)
    
    if verbose:
        print(f"\nOutputs:")
        print(f"  {csv_path}")
        if savefig:
            print(f"  {plot_path}")
    
    return corrected


def batch_jump_correction(
    data_dict: Dict[str, pd.Series],
    jumps_dict: Dict[str, List[str]],
    output_dir: str = '02_jump_corrected',
    savefig: bool = False,
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """
    Process multiple stations in batch mode.
    
    Args:
        data_dict: Dictionary mapping station names to time series
        jumps_dict: Dictionary mapping station names to jump dates
        output_dir: Output directory
        savefig: Save diagnostic plots
        verbose: Print detailed progress
    
    Returns:
        Dictionary of corrected time series keyed by station name
    """
    corrected_all = {}
    n_total = len(data_dict)
    
    print(f"\nCorrecting jumps for {n_total} stations...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    for i, (station_name, timeseries) in enumerate(data_dict.items(), 1):
        try:
            jump_dates = jumps_dict.get(station_name, [])
            
            corrected = run_jump_correction(
                timeseries=timeseries,
                station_name=station_name,
                jump_dates=jump_dates,
                output_dir=output_dir,
                savefig=savefig,
                verbose=verbose
            )
            corrected_all[station_name] = corrected
            
            # Progress indicator
            if not verbose:
                status = f"[{i}/{n_total}] {station_name}: {len(jump_dates)} jumps corrected"
                print(status)
                
        except Exception as e:
            print(f"[{i}/{n_total}] {station_name}: ERROR - {str(e)}")
            corrected_all[station_name] = timeseries  # Return original on error
    
    # Summary
    print("-" * 60)
    total_jumps = sum(len(jumps_dict.get(stn, [])) for stn in data_dict.keys())
    print(f"\nBatch Summary:")
    print(f"  Processed: {len(corrected_all)}/{n_total} stations")
    print(f"  Total jumps corrected: {total_jumps}")
    print(f"\nResults saved to: {output_dir}")
    
    return corrected_all