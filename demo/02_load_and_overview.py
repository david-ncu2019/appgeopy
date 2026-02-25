"""
Load and explore the groundwater dataset.

Loads from HDF5 and Excel, displays station locations on a map,
prints summary statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from appgeopy.core import TableArray, SpatialTableArray
from appgeopy.gwatertools import hdf5_to_data_dict, show_dictkeys_recursive
from appgeopy.visualize import save_figure, configure_axis
import h5py


OUTPUT_DIR = Path(__file__).parent
HDF5_FILE = OUTPUT_DIR / 'groundwater_data.h5'
EXCEL_FILE = OUTPUT_DIR / 'groundwater_data.xlsx'
FIGS_DIR = OUTPUT_DIR / 'figs'


def load_hdf5():
    """Load and display HDF5 structure."""
    print("\n" + "="*60)
    print("HDF5 File Structure")
    print("="*60)

    with h5py.File(HDF5_FILE, 'r') as f:
        show_dictkeys_recursive(hdf5_to_data_dict(f))

    print()


def load_excel_sheets():
    """Load Excel and print summary for each sheet."""
    print("="*60)
    print("Excel Sheets")
    print("="*60)

    sheet_names = TableArray.get_sheet_names(str(EXCEL_FILE))
    print(f"Found sheets: {sheet_names}\n")

    for sheet_name in sheet_names:
        ta = TableArray.from_excel(str(EXCEL_FILE), sheet_name=sheet_name)
        print(f"{sheet_name}:")
        print(f"  Rows: {len(ta)}, Columns: {list(ta.columns)}")

    print()


def create_station_map():
    """Create and save station location map."""
    print("="*60)
    print("Station Map")
    print("="*60)

    # Load metadata from HDF5
    with h5py.File(HDF5_FILE, 'r') as f:
        stations = []
        lons = []
        lats = []
        descs = []

        for station_id in sorted(f.keys()):
            group = f[station_id]
            stations.append(station_id)
            lons.append(group.attrs['lon'])
            lats.append(group.attrs['lat'])
            # Decode description if it's bytes
            desc = group.attrs['description']
            if isinstance(desc, bytes):
                desc = desc.decode()
            descs.append(desc)

    # Create GeoDataFrame
    df = pd.DataFrame({
        'station': stations,
        'lon': lons,
        'lat': lats,
        'description': descs,
    })

    spatial = SpatialTableArray.from_dataframe(
        df, x_col='lon', y_col='lat', crs='EPSG:4326'
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    spatial.plot(ax=ax, alpha=0.6, edgecolor='k', markersize=200, color='steelblue')

    # Annotate station names
    for idx, row in spatial.iterrows():
        ax.annotate(
            row['station'],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
        )

    configure_axis(
        ax,
        xlabel='Longitude (°E)',
        ylabel='Latitude (°N)',
        title='Groundwater Monitoring Station Locations',
        scaling_factor=0.8,
        fontsize_base=14,
    )
    ax.grid(True, alpha=0.3)

    # Save
    save_figure(fig, FIGS_DIR / 'station_map.png', dpi=300)
    print(f"[OK] Saved map: {FIGS_DIR / 'station_map.png'}")
    plt.close()


def print_statistics():
    """Print basic statistics for each station."""
    print("\n" + "="*60)
    print("Data Statistics")
    print("="*60)

    with h5py.File(HDF5_FILE, 'r') as f:
        for station_id in sorted(f.keys()):
            group = f[station_id]
            print(f"\n{station_id} ({group.attrs['description']}):")

            for well_name in ['shallow_well', 'mid_well', 'deep_well']:
                data = group[well_name][:]
                n_valid = (~np.isnan(data)).sum()
                n_missing = np.isnan(data).sum()
                pct_missing = 100 * n_missing / len(data)

                print(f"  {well_name}:")
                print(f"    Valid: {n_valid}/{len(data)} ({100-pct_missing:.1f}%)")
                print(f"    Mean: {np.nanmean(data):.4f} m")
                print(f"    Std: {np.nanstd(data):.4f} m")


def main():
    """Load and explore dataset."""
    print("\n" + "="*60)
    print("Groundwater Dataset Overview")
    print("="*60)

    # Check files exist
    if not HDF5_FILE.exists():
        print(f"✗ HDF5 file not found. Run 01_generate_dataset.py first.")
        return

    if not EXCEL_FILE.exists():
        print(f"✗ Excel file not found. Run 01_generate_dataset.py first.")
        return

    # Load and display
    load_hdf5()
    load_excel_sheets()
    print_statistics()
    create_station_map()

    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == '__main__':
    import numpy as np
    main()
