# appgeopy Demo Programs

This folder demonstrates the complete groundwater data analysis workflow using the appgeopy package.

## Dataset

**5 groundwater monitoring stations (GW01-GW05)** in SE Asia, each with 3 wells:
- `shallow_well` (aquifer zone 1)
- `mid_well` (aquifer zone 2)
- `deep_well` (aquifer zone 3)

**Time period:** 1 year (2023-01-01 to 2023-12-31), daily measurements

**Synthetic artifacts injected:**
- Linear trend: -0.002 m/day (groundwater depletion)
- Seasonality: Annual (0.8m amplitude) + semi-annual (0.3m amplitude) patterns
- Gaussian noise: variance = 0.02
- Equipment jumps: 1-2 per series, magnitude = 5× mean absolute value
- Random gaps: ~5% missing values

## Programs

### 1. `01_generate_dataset.py`

**Purpose:** Create synthetic groundwater dataset and save to multiple formats.

**Output:**
- `groundwater_data.h5` — HDF5 file with structure: `/{station_id}/{well_name}`
- `groundwater_data.xlsx` — Excel file with one sheet per station

**Run:**
```bash
python demo/01_generate_dataset.py
```

### 2. `02_load_and_overview.py`

**Purpose:** Load and explore the dataset; demonstrate SpatialTableArray.

**Output:**
- Console summary of HDF5 structure and statistics for each station
- `figs/station_map.png` — Geographic map of monitoring stations

**Run:**
```bash
python demo/02_load_and_overview.py
```

### 3. `03_analysis_pipeline.py`

**Purpose:** Full processing pipeline on a selected timeseries (GW03/shallow_well).

**Pipeline Steps:**
1. Jump detection (ruptures.Pelt algorithm)
2. Jump correction (anchor-segment approach)
3. Outlier removal (MAD-based threshold)
4. Gap filling (SSA - Singular Spectrum Analysis)
5. Smoothing (moving average, window=15)
6. Trend extraction (linear regression)
7. Seasonality detection (FFT-based)
8. Sinusoidal model fitting (Fourier components)

**Output:**
- Console: Jump locations, correction summary, trend slope, seasonal components, model fit metrics
- `figs/analysis_pipeline.png` — 4-panel diagnostic figure:
  - Panel 1: Raw signal with detected jump markers
  - Panel 2: Progressive processing (correction → gap fill → smoothing)
  - Panel 3: Trend line extraction
  - Panel 4: Sinusoidal model fit vs. observations

**Run:**
```bash
python demo/03_analysis_pipeline.py
```

## Example Results

From GW03/shallow_well analysis:
- **Jumps detected:** 2 equipment changes
- **Jumps corrected:** 2
- **Outliers removed:** 5 values
- **Trend slope:** -0.006446 m/day (depletion rate)
- **Model R²:** 0.545 (good fit)

## Key API Demonstrations

| Class/Function | Demo Used In |
|---|---|
| `TimeSeries.synthetic()` | Program 1 |
| `TimeSeries.detect_jumps()` | Program 3 |
| `TimeSeries.correct_jumps()` | Program 3 |
| `TimeSeries.fill_gaps()` | Program 3 |
| `TimeSeries.smooth()` | Program 3 |
| `TimeSeries.get_trend()` | Program 3 |
| `TimeSeries.find_seasonality()` | Program 3 |
| `TimeSeries.fit_sinusoidal()` | Program 3 |
| `TableArray.to_excel_sheet()` | Program 1 |
| `SpatialTableArray.from_dataframe()` | Program 2 |
| `gwatertools.data_to_hdf5()` | Program 1 |
| `visualize.configure_axis()` | Programs 2, 3 |

## File Structure

```
demo/
├── 01_generate_dataset.py          # Data generation
├── 02_load_and_overview.py         # Data exploration
├── 03_analysis_pipeline.py         # Analysis & visualization
├── README.md                        # This file
├── groundwater_data.h5             # Synthetic HDF5 dataset
├── groundwater_data.xlsx           # Synthetic Excel dataset
└── figs/
    ├── station_map.png             # Station locations
    └── analysis_pipeline.png        # Analysis results
```

## Running All Programs

```bash
# Generate dataset
python demo/01_generate_dataset.py

# Explore data
python demo/02_load_and_overview.py

# Run full analysis
python demo/03_analysis_pipeline.py

# View figures
# Open demo/figs/station_map.png and demo/figs/analysis_pipeline.png
```

## Notes

- All programs use relative paths, so run from the repo root or adjust paths as needed.
- Programs are designed to be independent; you can run any one after dataset generation.
- Modify the `TARGET_STATION` and `TARGET_WELL` variables in `03_analysis_pipeline.py` to analyze different timeseries.
- For production use, adapt programs to load real data via `TableArray.from_excel()`, `SpatialTableArray.from_csv()`, or `DataCube.from_netcdf()`.
