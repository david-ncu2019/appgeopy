"""
Full analysis pipeline on groundwater timeseries.

Selects GW03/shallow_well, performs:
  1. Jump detection & correction
  2. Outlier removal
  3. Gap filling (SSA)
  4. Smoothing
  5. Trend extraction
  6. Seasonality detection
  7. Sinusoidal model fit

Produces comprehensive 4-panel visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from pathlib import Path

from appgeopy.core import TimeSeries
from appgeopy.gwatertools import hdf5_to_data_dict
from appgeopy.visualize import save_figure, configure_axis
import matplotlib.dates as mdates


OUTPUT_DIR = Path(__file__).parent
HDF5_FILE = OUTPUT_DIR / 'groundwater_data.h5'
FIGS_DIR = OUTPUT_DIR / 'figs'

# Select station and well
TARGET_STATION = 'GW03'
TARGET_WELL = 'shallow_well'


def load_timeseries() -> TimeSeries:
    """Load target timeseries from HDF5."""
    with h5py.File(HDF5_FILE, 'r') as f:
        group = f[TARGET_STATION]
        date_bytes = group['dates'][:]
        # Decode bytes to strings for datetime parsing
        date_strings = [d.decode() if isinstance(d, bytes) else d for d in date_bytes]
        dates = pd.to_datetime(date_strings)
        values = group[TARGET_WELL][:]

    ts = TimeSeries(values, index=dates, name=TARGET_WELL)
    ts = ts.align_to_fulltime(freq='D')
    return ts


def run_analysis_pipeline(ts: TimeSeries) -> dict:
    """
    Run full processing pipeline, return results dict.
    """
    results = {}

    print("\n" + "="*60)
    print(f"Analysis: {TARGET_STATION} / {TARGET_WELL}")
    print("="*60)

    # 1. Jump detection
    print("\n1. Jump Detection...", end=' ')
    jump_dates, jump_types, gaps = ts.detect_jumps(
        penalty=25,
        min_segment_days=60,
        gap_threshold_days=60
    )
    real_jumps = [d for d, t in zip(jump_dates, jump_types) if t == 'equipment_change']
    print(f"Found {len(real_jumps)} real jumps")
    results['jump_dates'] = real_jumps
    results['jumps_visualization'] = (jump_dates, jump_types)

    # 2. Jump correction
    print("2. Jump Correction...", end=' ')
    if real_jumps:
        corrected, steps = ts.correct_jumps(real_jumps)
        print(f"Corrected {len(steps)} jumps")
    else:
        corrected = ts.copy()
        steps = []
        print("No jumps to correct (OK)")
    results['corrected'] = corrected

    # 3. Outlier detection & removal
    print("3. Outlier Detection...", end=' ')
    cleaned = corrected.detect_outliers(threshold=3.5, use_time=True)
    n_outliers = (~corrected.isna()).sum() - (~cleaned.isna()).sum()
    print(f"Removed {n_outliers} outliers")
    results['cleaned'] = cleaned

    # 4. Gap filling (SSA)
    print("4. Gap Filling (SSA)...", end=' ')
    try:
        filled = cleaned.fill_gaps(
            method='ssa',
            variance_threshold=0.9,
            max_iter=30,
        )
        print("OK")
    except ValueError as e:
        print(f"Skipped (too many gaps)")
        filled = cleaned.interpolate(method='linear')
    results['filled'] = filled

    # 5. Smoothing
    print("5. Smoothing (window=15)...", end=' ')
    smoothed = filled.smooth(window=15)
    print("OK")
    results['smoothed'] = smoothed

    # 6. Trend extraction
    print("6. Trend Analysis...", end=' ')
    trend, slope = smoothed.get_trend(method='linear')
    print(f"Slope = {slope:.6f} m/day")
    results['trend'] = trend
    results['slope'] = slope

    # 7. Seasonality
    print("7. Seasonality Detection...", end=' ')
    seasonality_df = smoothed.find_seasonality()
    print(f"Found {len(seasonality_df)} components")
    results['seasonality'] = seasonality_df

    # 8. Sinusoidal fit
    print("8. Sinusoidal Model Fit...", end=' ')
    try:
        estimation, params = smoothed.fit_sinusoidal(periods=[365.25, 182.6])
        results['estimation'] = estimation
        results['model_params'] = params
        print("OK")
    except Exception as e:
        print(f"Skipped (error: {str(e)[:50]})")
        results['estimation'] = None
        results['model_params'] = None

    # 9. Metrics
    print("9. Model Evaluation...", end=' ')
    if results['estimation'] is not None:
        metrics = smoothed.evaluate_fit(results['estimation'])
        results['metrics'] = metrics
        print(f"R² = {metrics['R2']:.4f}")
    else:
        results['metrics'] = None
        print("Skipped")

    return results


def plot_analysis(ts: TimeSeries, results: dict):
    """Create 4-panel diagnostic figure."""
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ========== Panel 1: Raw signal + detected jumps ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts.index, ts.values, 'o-', label='Raw data', linewidth=1.5,
             markersize=3, alpha=0.7, color='navy')

    # Mark detected jumps
    if results['jumps_visualization'][0]:
        for jdate, jtype in zip(*results['jumps_visualization']):
            color = 'red' if jtype == 'equipment_change' else 'orange'
            ax1.axvline(jdate, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    configure_axis(ax1, ylabel='Water Level (m)', title='1. Raw Signal + Detected Jumps',
                   scaling_factor=0.8, fontsize_base=12)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ========== Panel 2: Processing steps (corrected + filled + smoothed) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts.index, results['corrected'].values, 'o-', label='After correction',
             linewidth=1.5, markersize=2, alpha=0.6, color='orange')
    ax2.plot(results['filled'].index, results['filled'].values, 'o-', label='After gap fill',
             linewidth=1.5, markersize=2, alpha=0.6, color='green')
    ax2.plot(results['smoothed'].index, results['smoothed'].values, '-', label='After smoothing',
             linewidth=2, color='darkgreen')

    configure_axis(ax2, ylabel='Water Level (m)', title='2. Progressive Processing',
                   scaling_factor=0.8, fontsize_base=12)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ========== Panel 3: Trend line ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['smoothed'].index, results['smoothed'].values, 'o-',
             label='Smoothed signal', linewidth=1.5, markersize=3, alpha=0.7, color='steelblue')
    ax3.plot(results['trend'].index, results['trend'].values, '-',
             label=f'Linear trend (slope={results["slope"]:.6f} m/day)',
             linewidth=2.5, color='red')

    configure_axis(ax3, ylabel='Water Level (m)', title='3. Trend Extraction',
                   scaling_factor=0.8, fontsize_base=12)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ========== Panel 4: Sinusoidal model ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['smoothed'].index, results['smoothed'].values, 'o-',
             label='Observed (smoothed)', linewidth=1.5, markersize=3, alpha=0.7, color='steelblue')

    if results['estimation'] is not None:
        ax4.plot(results['smoothed'].index, results['estimation'], '-',
                 label=f"Model fit (R²={results['metrics']['R2']:.4f})",
                 linewidth=2.5, color='darkred')
        ax4.legend(loc='upper left', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Model fit not available', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=11, style='italic', color='gray')

    configure_axis(ax4, ylabel='Water Level (m)', title='4. Sinusoidal Model Fit',
                   scaling_factor=0.8, fontsize_base=12)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Groundwater Analysis Pipeline: {TARGET_STATION} / {TARGET_WELL}',
                 fontsize=14, fontweight='bold', y=0.995)

    return fig


def print_results(results: dict):
    """Print summary of analysis results."""
    print("\n" + "="*60)
    print("Analysis Results Summary")
    print("="*60)

    print(f"\nTarget: {TARGET_STATION} / {TARGET_WELL}")
    print(f"Real jumps corrected: {len(results['jump_dates'])}")
    print(f"Linear trend slope: {results['slope']:.6f} m/day")

    if results['seasonality'] is not None and len(results['seasonality']) > 0:
        print(f"\nDominant seasonal components:")
        # Get the first few rows and print whatever columns are available
        for idx, row in results['seasonality'].head(5).iterrows():
            cols = results['seasonality'].columns
            if 'Period' in cols and 'Amplitude' in cols:
                print(f"  Period {row['Period']:.1f} days | Amplitude {row['Amplitude']:.4f} m")
            else:
                # Print first 2 columns as fallback
                print(f"  {row.iloc[0]:.4f} | {row.iloc[1]:.4f}")

    if results['metrics'] is not None:
        print(f"\nModel fit metrics:")
        for key, val in results['metrics'].items():
            print(f"  {key}: {val:.6f}")


def main():
    """Run full pipeline."""
    print("\n" + "="*60)
    print("Groundwater Data Analysis Pipeline")
    print("="*60)

    # Check HDF5 exists
    if not HDF5_FILE.exists():
        print(f"✗ HDF5 file not found. Run 01_generate_dataset.py first.")
        return

    # Load data
    print(f"\nLoading {TARGET_STATION} / {TARGET_WELL}...")
    ts = load_timeseries()
    print(f"  Length: {len(ts)} days")
    print(f"  Missing: {ts.isna().sum()} ({100*ts.isna().sum()/len(ts):.1f}%)")

    # Run pipeline
    results = run_analysis_pipeline(ts)

    # Print results
    print_results(results)

    # Create and save figure
    print("\nCreating visualization...", end=' ')
    fig = plot_analysis(ts, results)
    # Use matplotlib directly to avoid helper function issues
    output_path = FIGS_DIR / 'analysis_pipeline.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
