"""
GPS Jump Detection Module
Automatically detects potential jumps with batch processing support.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def detect_data_gaps(
    timeseries: pd.Series, 
    gap_threshold_days: int = 30
) -> List[Dict[str, any]]:
    """Identify large data gaps between valid measurements."""
    valid_indices = timeseries.dropna().index
    
    gaps = []
    for i in range(len(valid_indices) - 1):
        gap_days = (valid_indices[i+1] - valid_indices[i]).days
        if gap_days > gap_threshold_days:
            gaps.append({
                'start': str(valid_indices[i].date()),
                'end': str(valid_indices[i+1].date()),
                'days': int(gap_days),
            })
    
    return gaps


def detect_jumps_smart(
    timeseries: pd.Series,
    penalty: int = 20,
    min_segment_days: int = 90,
    gap_threshold_days: int = 60
) -> Tuple[List[pd.Timestamp], List[str], List[Dict]]:
    """
    Smart jump detection that filters out gap boundaries.
    
    Args:
        timeseries: Time series with DatetimeIndex
        penalty: Ruptures penalty (15-30, higher = fewer jumps)
        min_segment_days: Minimum days between jumps
        gap_threshold_days: Gap threshold for filtering
    
    Returns:
        jump_dates: Detected jump timestamps
        jump_types: Jump classifications
        gaps: Data gap records
    """
    valid_data = timeseries.dropna()
    
    if len(valid_data) < 2 * min_segment_days:
        return [], [], []
    
    try:
        # Downsample for large datasets
        if len(valid_data) > 2000:
            downsample_factor = len(valid_data) // 2000
            downsampled_values = valid_data.iloc[::downsample_factor].values
            downsampled_index = valid_data.index[::downsample_factor]
            
            signal_data = downsampled_values.reshape(-1, 1)
            algo = rpt.Pelt(
                model='rbf',
                min_size=min_segment_days // downsample_factor,
                jump=1
            ).fit(signal_data)
            changepoints_down = algo.predict(pen=penalty)
            
            jump_dates_candidates = [
                downsampled_index[cp - 1] for cp in changepoints_down[:-1]
            ]
            
            # Refine in ±30 day window
            jump_dates_refined = []
            for approx_date in jump_dates_candidates:
                window_start = approx_date - pd.Timedelta(days=30)
                window_end = approx_date + pd.Timedelta(days=30)
                window_data = valid_data[
                    (valid_data.index >= window_start) & 
                    (valid_data.index <= window_end)
                ]
                
                if len(window_data) > 20:
                    diffs = np.abs(np.diff(window_data.values))
                    max_diff_idx = np.argmax(diffs)
                    refined_date = window_data.index[max_diff_idx]
                    jump_dates_refined.append(refined_date)
                else:
                    jump_dates_refined.append(approx_date)
        else:
            # Small dataset
            signal_data = valid_data.values.reshape(-1, 1)
            algo = rpt.Pelt(
                model='rbf',
                min_size=min_segment_days,
                jump=1
            ).fit(signal_data)
            changepoints = algo.predict(pen=penalty)
            
            jump_indices_in_valid = [cp - 1 for cp in changepoints[:-1]]
            jump_dates_refined = [valid_data.index[idx] for idx in jump_indices_in_valid]
        
        # Identify gaps
        gaps = detect_data_gaps(timeseries, gap_threshold_days=gap_threshold_days)
        
        # Filter out gap boundaries
        jump_dates_filtered = []
        jump_types = []
        
        for date in jump_dates_refined:
            is_gap_boundary = False
            
            for gap in gaps:
                gap_start = pd.to_datetime(gap['start'])
                gap_end = pd.to_datetime(gap['end'])
                days_from_gap_start = abs((date - gap_start).days)
                days_from_gap_end = abs((date - gap_end).days)
                
                if days_from_gap_start < 30 or days_from_gap_end < 30:
                    is_gap_boundary = True
                    break
            
            jump_dates_filtered.append(date)
            jump_types.append('gap_boundary' if is_gap_boundary else 'equipment_change')
        
        return jump_dates_filtered, jump_types, gaps
        
    except Exception as e:
        if __debug__:
            print(f"Detection error: {e}")
        return [], [], []


def plot_jump_detection(
    timeseries: pd.Series,
    jump_dates: List[pd.Timestamp],
    jump_types: List[str],
    gaps: List[Dict],
    save_path: Path
) -> None:
    """Create diagnostic plot for jump detection."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot data
    ax.plot(timeseries, 'o', ms=2, alpha=0.5, color='steelblue', label='Data')
    
    # Plot gaps
    for gap in gaps:
        gap_start = pd.to_datetime(gap['start'])
        gap_end = pd.to_datetime(gap['end'])
        ax.axvspan(gap_start, gap_end, alpha=0.2, color='gray', label='_nolegend_')
    
    # Filter jumps
    equipment_jumps = [jd for jd, jt in zip(jump_dates, jump_types) if jt == 'equipment_change']
    gap_jumps = [jd for jd, jt in zip(jump_dates, jump_types) if jt == 'gap_boundary']
    
    trans = ax.get_xaxis_transform()
    
    # Plot equipment jumps with labels
    for jd in equipment_jumps:
        ax.axvline(jd, color='red', ls='--', lw=2.5, alpha=0.8, label='_nolegend_')
        ax.text(
            x=jd, y=1.01, s=jd.strftime('%Y-%m-%d'),
            transform=trans, color='red', rotation=45,
            ha='left', va='bottom', fontsize=10, fontweight='bold'
        )
    
    # Plot gap jumps
    for jd in gap_jumps:
        ax.axvline(jd, color='orange', ls=':', lw=2, alpha=0.6, label='_nolegend_')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', marker='o', linestyle='None', markersize=5, label='GPS Data'),
        Line2D([0], [0], color='red', linestyle='--', lw=2.5, label='Likely Equipment Change'),
        Line2D([0], [0], color='orange', linestyle=':', lw=2, label='Likely Gap Boundary'),
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.2, label='Data Gap'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax.set_ylabel('Displacement (m)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_jump_detection(
    timeseries: pd.Series,
    station_name: str,
    output_dir: str = '01_jump_detection',
    penalty: int = 20,
    min_segment_days: int = 90,
    gap_threshold_days: int = 60,
    savefig: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run jump detection and save results.
    
    Args:
        timeseries: Time series with DatetimeIndex
        station_name: Station identifier
        output_dir: Output directory
        penalty: Ruptures penalty parameter
        min_segment_days: Minimum days between jumps
        gap_threshold_days: Gap detection threshold
        savefig: Save diagnostic plot
        verbose: Print progress messages
    
    Returns:
        Detection results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"JUMP DETECTION: {station_name}")
        print(f"{'='*60}")
    
    # Detect jumps
    jump_dates, jump_types, gaps = detect_jumps_smart(
        timeseries,
        penalty=penalty,
        min_segment_days=min_segment_days,
        gap_threshold_days=gap_threshold_days
    )
    
    # Prepare results
    results = {
        'station': station_name,
        'detection_params': {
            'penalty': penalty,
            'min_segment_days': min_segment_days,
            'gap_threshold_days': gap_threshold_days,
        },
        'data_gaps': gaps,
        'detected_jumps': [
            {
                'date': str(jd.date()),
                'type': jt,
                'action': 'REVIEW' if jt == 'equipment_change' else 'LIKELY_FALSE',
            }
            for jd, jt in zip(jump_dates, jump_types)
        ],
    }
    
    # Save JSON
    json_path = output_path / f"{station_name}_jumps.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if savefig:
        plot_path = output_path / f"{station_name}_detection.png"
        plot_jump_detection(timeseries, jump_dates, jump_types, gaps, plot_path)
    
    if verbose:
        equipment_jumps = [
            j for j in results['detected_jumps'] 
            if j['type'] == 'equipment_change'
        ]
        
        print(f"Data gaps: {len(gaps)}")
        print(f"Detected jumps: {len(jump_dates)}")
        
        if equipment_jumps:
            print(f"\nLikely REAL jumps:")
            for j in equipment_jumps:
                print(f"  - {j['date']}")
        
        print(f"\nOutputs:")
        print(f"  {json_path}")
        if savefig:
            print(f"  {plot_path}")
        print(f"\n→ Review and select real jumps for next step")
    
    return results


def batch_jump_detection(
    data_dict: Dict[str, pd.Series],
    output_dir: str = '01_jump_detection',
    penalty: int = 20,
    min_segment_days: int = 90,
    gap_threshold_days: int = 60,
    savefig: bool = False,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Process multiple stations in batch mode.
    
    Args:
        data_dict: Dictionary mapping station names to time series
        output_dir: Output directory
        penalty: Ruptures penalty parameter
        min_segment_days: Minimum days between jumps
        gap_threshold_days: Gap detection threshold
        savefig: Save diagnostic plots
        verbose: Print detailed progress (not recommended for large batches)
    
    Returns:
        Dictionary of results keyed by station name
    """
    results_all = {}
    n_total = len(data_dict)
    
    print(f"\nProcessing {n_total} stations...")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: penalty={penalty}, min_segment_days={min_segment_days}, gap_threshold_days={gap_threshold_days}")
    print("-" * 60)
    
    for i, (station_name, timeseries) in enumerate(data_dict.items(), 1):
        try:
            results = run_jump_detection(
                timeseries=timeseries,
                station_name=station_name,
                output_dir=output_dir,
                penalty=penalty,
                min_segment_days=min_segment_days,
                gap_threshold_days=gap_threshold_days,
                savefig=savefig,
                verbose=verbose
            )
            results_all[station_name] = results
            
            # Progress indicator
            if not verbose:
                n_gaps = len(results['data_gaps'])
                n_jumps = len(results['detected_jumps'])
                equipment_jumps = sum(
                    1 for j in results['detected_jumps'] 
                    if j['type'] == 'equipment_change'
                )
                status = f"[{i}/{n_total}] {station_name}: {n_gaps} gaps, {n_jumps} jumps ({equipment_jumps} equipment)"
                print(status)
                
        except Exception as e:
            print(f"[{i}/{n_total}] {station_name}: ERROR - {str(e)}")
            results_all[station_name] = {'error': str(e)}
    
    # Summary
    print("-" * 60)
    n_success = sum(1 for r in results_all.values() if 'error' not in r)
    n_failed = n_total - n_success
    
    total_equipment_jumps = sum(
        sum(1 for j in r['detected_jumps'] if j['type'] == 'equipment_change')
        for r in results_all.values() if 'error' not in r
    )
    
    print(f"\nBatch Summary:")
    print(f"  Processed: {n_success}/{n_total} stations")
    print(f"  Failed: {n_failed}")
    print(f"  Total equipment jumps detected: {total_equipment_jumps}")
    print(f"\nResults saved to: {output_dir}")
    
    return results_all