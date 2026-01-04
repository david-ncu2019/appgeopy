"""
GPS Jump Correction Module (Robust Anchor Strategy)
Corrects time series by aligning all segments to the longest stable segment.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def correct_jumps(
    timeseries: pd.Series,
    jump_dates: List[str]
) -> Tuple[pd.Series, List[float]]:
    """
    Corrects jumps by aligning everything to the LONGEST segment (The Anchor).
    
    Args:
        timeseries: Time series with DatetimeIndex
        jump_dates: List of jump date strings (e.g., ['2022-06-17'])
    
    Returns:
        corrected: Jump-corrected time series
        step_sizes: List of step sizes at each jump date (for plotting)
    """
    corrected = timeseries.copy()
    
    # 1. Handle Input
    if not jump_dates:
        return corrected, []
    
    # Sort and unique dates
    jump_dates_dt = sorted(list(set([pd.to_datetime(d) for d in jump_dates])))
    
    # Find indices in the array
    jump_indices = []
    valid_jump_dates = []
    
    for date in jump_dates_dt:
        if date in timeseries.index:
            loc = timeseries.index.get_loc(date)
            # Handle non-unique index
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, np.ndarray):
                loc = np.where(loc)[0][0]
            
            # Only add if it's not at the very start or end
            if 0 < loc < len(timeseries):
                jump_indices.append(loc)
                valid_jump_dates.append(date)
    
    if not jump_indices:
        return corrected, []
    
    # 2. Define Segments
    # segments = [(start, end), (start, end), ...]
    segments = []
    start = 0
    for jump_idx in jump_indices:
        segments.append((start, jump_idx))
        start = jump_idx
    segments.append((start, len(timeseries)))
    
    # 3. Find the Anchor (Longest Segment)
    # We want to keep the longest stable period fixed at 0 offset.
    lengths = [end - start for start, end in segments]
    anchor_idx = np.argmax(lengths)
    
    # Initialize offsets for each segment
    segment_offsets = np.zeros(len(segments))
    step_sizes_map = {} # Key: jump_index, Value: step_size
    
    values = timeseries.values
    
    # 4. Backward Pass (Align segments BEFORE anchor)
    # e.g., If Anchor is Seg 1, we align Seg 0 to Seg 1.
    for i in range(anchor_idx - 1, -1, -1):
        # Current Segment: i
        # Target Segment (already aligned): i + 1
        
        curr_s, curr_e = segments[i]
        next_s, next_e = segments[i+1] # This is the target we align TO
        
        # Get Data at the "Edge" (Boundary between i and i+1)
        curr_vals = values[curr_s:curr_e]
        next_vals = values[next_s:next_e]
        
        curr_valid = curr_vals[~np.isnan(curr_vals)]
        next_valid = next_vals[~np.isnan(next_vals)]
        
        step = 0.0
        if len(curr_valid) >= 5 and len(next_valid) >= 5:
            n_edge = min(30, len(curr_valid)//2, len(next_valid)//2)
            
            # We want: Median(Corrected_Curr) == Median(Corrected_Next)
            # Median(Raw_Curr + Offset_Curr) == Median(Raw_Next + Offset_Next)
            # Offset_Curr = Median(Raw_Next + Offset_Next) - Median(Raw_Curr)
            
            target_level = np.median(next_valid[:n_edge] + segment_offsets[i+1])
            current_level = np.median(curr_valid[-n_edge:])
            
            # Offset needed for THIS segment
            segment_offsets[i] = target_level - current_level
            
            # The "Step Size" at the jump is the difference in offsets
            # Jump corresponds to boundary `next_s` (which is same as `curr_e`)
            step_sizes_map[next_s] = segment_offsets[i+1] - segment_offsets[i]
            
    # 5. Forward Pass (Align segments AFTER anchor)
    # e.g., If Anchor is Seg 1, we align Seg 2 to Seg 1.
    for i in range(anchor_idx + 1, len(segments)):
        # Current Segment: i
        # Previous Segment (already aligned): i - 1
        
        curr_s, curr_e = segments[i]
        prev_s, prev_e = segments[i-1]
        
        curr_vals = values[curr_s:curr_e]
        prev_vals = values[prev_s:prev_e]
        
        curr_valid = curr_vals[~np.isnan(curr_vals)]
        prev_valid = prev_vals[~np.isnan(prev_vals)]
        
        step = 0.0
        if len(curr_valid) >= 5 and len(prev_valid) >= 5:
            n_edge = min(30, len(curr_valid)//2, len(prev_valid)//2)
            
            # Target is the END of the previous segment
            target_level = np.median(prev_valid[-n_edge:] + segment_offsets[i-1])
            current_level = np.median(curr_valid[:n_edge])
            
            segment_offsets[i] = target_level - current_level
            
            # Jump corresponds to boundary `curr_s`
            step_sizes_map[curr_s] = segment_offsets[i-1] - segment_offsets[i]

    # 6. Apply Offsets
    for i, (start, end) in enumerate(segments):
        if segment_offsets[i] != 0:
            corrected.iloc[start:end] += segment_offsets[i]
            
    # 7. Collect Step Sizes in order of jump_dates
    # Note: step_size is (Pre - Post), consistent with standard definition
    final_steps = []
    for idx in jump_indices:
        final_steps.append(step_sizes_map.get(idx, 0.0))
        
    return corrected, final_steps


def plot_jump_correction(
    original: pd.Series,
    corrected: pd.Series,
    jump_dates: List[str],
    offsets: List[float],
    save_path: Path
) -> None:
    """Publication-quality plot."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300})

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0.1})
    
    # Valid jump dates used
    jump_dates_dt = sorted(list(set([pd.to_datetime(d) for d in jump_dates])))
    jump_dates_dt = [d for d in jump_dates_dt if d in original.index]

    # Plot 1: Original
    ax0 = axes[0]
    ax0.plot(original.index, original.values, '.', ms=2, color='gray', alpha=0.5, label='Raw')
    
    for jd, step in zip(jump_dates_dt, offsets):
        ax0.axvline(jd, color='red', lw=1, ls='--')
        # Annotate step
        ax0.text(jd, ax0.get_ylim()[1], f" {step*1000:+.1f}mm", 
                 rotation=90, verticalalignment='top', color='red', fontsize=8, fontweight='bold')

    ax0.set_ylabel('Original (m)')
    ax0.legend(loc='upper right')
    ax0.grid(True, ls=':', alpha=0.5)

    # Plot 2: Corrected
    ax1 = axes[1]
    ax1.plot(corrected.index, corrected.values, '.', ms=2, color='teal', alpha=0.5, label='Corrected')
    ax1.set_ylabel('Corrected (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def run_jump_correction(
    timeseries: pd.Series,
    station_name: str,
    jump_dates: List[str],
    output_dir: str = '02_jump_corrected',
    savefig: bool = False,
    verbose: bool = True
) -> pd.Series:
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"JUMP CORRECTION: {station_name}")
        print(f"{'='*60}")
        
    corrected, steps = correct_jumps(timeseries, jump_dates)
    
    if verbose:
        print(f"Jumps: {len(steps)}")
        for i, (d, s) in enumerate(zip(jump_dates, steps)):
            print(f"  {i+1}. {d}: {s*1000:+.1f} mm")
            
    # Save
    csv_name = output_path / f"{station_name}_corrected.csv"
    corrected.to_csv(csv_name, header=True)
    
    if savefig:
        plot_name = output_path / f"{station_name}_correction.png"
        plot_jump_correction(timeseries, corrected, jump_dates, steps, plot_name)
        
    if verbose:
        print(f"\nOutputs:\n  {csv_name}")
    
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