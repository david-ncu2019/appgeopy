"""
Generate synthetic groundwater level dataset with realistic artifacts.

Creates 5 stations (GW01-GW05) with 3 wells each (shallow, mid, deep).
Injects jumps, gaps, and noise. Saves to HDF5 and Excel.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path

from appgeopy.core import TimeSeries, TableArray
from appgeopy.gwatertools import data_to_hdf5, metadata_to_hdf5


# Station metadata (lon, lat, description)
STATIONS = {
    'GW01': {'lon': 100.5, 'lat': 13.8, 'desc': 'Bangkok urban zone'},
    'GW02': {'lon': 101.2, 'lat': 14.4, 'desc': 'Central plain north'},
    'GW03': {'lon': 99.8,  'lat': 12.9, 'desc': 'Coastal station'},
    'GW04': {'lon': 102.0, 'lat': 15.1, 'desc': 'Plateau region'},
    'GW05': {'lon': 100.0, 'lat': 14.0, 'desc': 'Agricultural area'},
}

WELLS = ['shallow_well', 'mid_well', 'deep_well']

OUTPUT_DIR = Path(__file__).parent
HDF5_FILE = OUTPUT_DIR / 'groundwater_data.h5'
EXCEL_FILE = OUTPUT_DIR / 'groundwater_data.xlsx'


def generate_station_data(station_id: str, seed_offset: int) -> dict:
    """
    Generate 3 timeseries (shallow/mid/deep wells) for one station.

    Each series has:
    - Annual (0.8m) + semi-annual (0.3m) seasonal patterns
    - Linear downward trend (-0.002 m/day)
    - Gaussian noise (variance 0.02)
    - 1-2 random jumps (magnitude = 5× mean absolute value)
    - ~5% random gaps (NaN)
    """
    station_data = {}

    for well_idx, well_name in enumerate(WELLS):
        seed = seed_offset + well_idx

        # Generate base synthetic signal
        ts = TimeSeries.synthetic(
            start_date='2023-01-01',
            end_date='2023-12-31',
            amplitude_list=[0.8, 0.3],
            period_list=[1.0, 0.5],  # annual, semi-annual (in years)
            linear_slope=-0.002,      # m/day
            variance=0.02,
            random_seed=seed
        )

        # Inject 1-2 jumps
        n_jumps = np.random.RandomState(seed + 100).randint(1, 3)
        mean_abs_val = ts.abs().mean()
        jump_mag = 5.0 * mean_abs_val

        # Convert to numpy array for mutation
        ts_values = ts.values.copy()
        n_points = len(ts_values)

        for j in range(n_jumps):
            jump_pos = np.random.RandomState(seed + 200 + j).randint(50, n_points - 50)
            jump_sign = np.random.RandomState(seed + 300 + j).choice([-1, 1])
            ts_values[jump_pos:] += jump_sign * jump_mag

        # Create new TimeSeries with jumps
        ts_with_jumps = TimeSeries(ts_values, index=ts.index, name=well_name)

        # Inject gaps (~5% missing)
        ts_with_gaps = ts_with_jumps.copy()
        gap_mask = np.random.RandomState(seed + 400).random(len(ts_with_gaps)) < 0.05
        ts_with_gaps.loc[gap_mask] = np.nan

        station_data[well_name] = ts_with_gaps

    return station_data


def save_hdf5(all_data: dict, stations_meta: dict):
    """Save all station data to HDF5 with structure: /{station_id}/{well_name}"""
    with h5py.File(HDF5_FILE, 'w') as f:
        for station_id, wells_dict in all_data.items():
            group = f.create_group(station_id)

            # Save dates once per station (all wells share same index)
            dates = wells_dict[WELLS[0]].index
            # Store dates as ISO format strings for HDF5 compatibility
            date_strings = [d.strftime('%Y-%m-%d') for d in dates]
            group.create_dataset('dates', data=date_strings, dtype=h5py.string_dtype())

            # Save each well
            for well_name, ts in wells_dict.items():
                group.create_dataset(well_name, data=ts.values)

            # Save metadata as attributes
            meta = stations_meta[station_id]
            group.attrs['lon'] = meta['lon']
            group.attrs['lat'] = meta['lat']
            group.attrs['description'] = meta['desc']

    print(f"[OK] Saved HDF5: {HDF5_FILE}")


def save_excel(all_data: dict):
    """Save each station as a sheet in Excel file."""
    for idx, (station_id, wells_dict) in enumerate(all_data.items()):
        # Get date index from first well
        date_index = wells_dict[WELLS[0]].index

        # Build DataFrame: date + all wells
        df_dict = {'date': date_index}
        for well_name in WELLS:
            df_dict[well_name] = wells_dict[well_name].values

        df = pd.DataFrame(df_dict)
        ta = TableArray(df)

        # First sheet: mode='w', others: mode='a'
        mode = 'w' if idx == 0 else 'a'
        ta.to_excel_sheet(
            str(EXCEL_FILE),
            sheet_name=station_id,
            mode=mode,
            if_sheet_exists='replace',
            index=False,
            verbose=False
        )

    print(f"[OK] Saved Excel: {EXCEL_FILE}")


def main():
    """Generate full dataset."""
    print("=" * 60)
    print("Groundwater Dataset Generation")
    print("=" * 60)

    all_data = {}

    # Generate data for all stations
    for station_idx, station_id in enumerate(sorted(STATIONS.keys())):
        print(f"\nGenerating {station_id}...", end=' ')
        seed_offset = station_idx * 1000
        station_data = generate_station_data(station_id, seed_offset)
        all_data[station_id] = station_data
        print("OK")

    # Save to both formats
    print("\nSaving to disk...")
    save_hdf5(all_data, STATIONS)
    save_excel(all_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Stations: {list(sorted(STATIONS.keys()))}")
    print(f"Wells per station: {WELLS}")
    print(f"Date range: 2023-01-01 to 2023-12-31 (365 days)")
    print(f"Total timeseries: {len(STATIONS) * len(WELLS)}")

    # Show stats for one example
    example_ts = all_data['GW01']['shallow_well']
    n_missing = example_ts.isna().sum()
    print(f"\nExample (GW01/shallow_well):")
    print(f"  - Length: {len(example_ts)}")
    print(f"  - Missing values: {n_missing} ({100*n_missing/len(example_ts):.1f}%)")
    print(f"  - Mean: {example_ts.mean():.4f} m")
    print(f"  - Std: {example_ts.std():.4f} m")


if __name__ == '__main__':
    main()
