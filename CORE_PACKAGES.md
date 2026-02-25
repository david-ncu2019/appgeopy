# Core Packages Guide

## Overview

The **core** package introduces a modern, class-based API for appgeopy, built on top of well-established data structures from pandas, geopandas, and xarray. This replaces the previous flat function-based approach with **chainable, object-oriented methods** that are more intuitive and powerful.

---

## Architecture

### Class Hierarchy

```
TimeSeries (pd.Series)
├── 1D time-indexed data
├── Subclasses pandas.Series
└── Preserves processing history

TableArray (pd.DataFrame)
├── Tabular data with metadata
├── Subclasses pandas.DataFrame
└── Excel/CSV/JSON I/O

SpatialTableArray (gpd.GeoDataFrame)
├── Geospatial tabular data
├── Subclasses geopandas.GeoDataFrame
└── Shapefile/GeoJSON/GeoPackage/KML I/O

DataCube (xr.Dataset wrapper)
├── Regular gridded 3D arrays (time × x × y)
├── Wraps xarray.Dataset (not inheritance)
└── NetCDF/Zarr I/O
```

---

## 1. TimeSeries

A 1D time-indexed data container with integrated analysis and preprocessing methods.

### Import

```python
from appgeopy.core import TimeSeries
```

### Construction

```python
import pandas as pd
import numpy as np

# From raw data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = np.random.randn(365)
ts = TimeSeries(values, index=dates, name='displacement')

# From existing pandas.Series
s = pd.Series(values, index=dates)
ts = TimeSeries(s)

# Generate synthetic data
ts = TimeSeries.synthetic(
    start_date='2020-01-01',
    end_date='2023-12-31',
    amplitude_list=[1.0, 0.5],        # Multiple seasonal components
    period_list=[1.0, 0.5],            # In years
    linear_slope=0.001,                # Trend per day
    variance=0.01,                     # Noise level
    random_seed=42
)
```

### Key Methods

#### Data Alignment
```python
# Ensure complete date range (insert NaNs for missing dates)
ts_full = ts.align_to_fulltime(freq='D')

# Find common dates with another series
common_idx = ts.intersect_index(other_ts.index)

# Get numeric indices for regression (non-NaN positions)
x = ts.numeric_index()
```

#### Jump Detection & Correction
```python
# Detect equipment changes and data gaps
jump_dates, jump_types, gaps = ts.detect_jumps(
    penalty=25,              # Higher = fewer jumps (range 15-30)
    min_segment_days=90,     # Minimum segment between jumps
    gap_threshold_days=60    # Gaps larger than this are ignored
)

# Filter for real equipment changes (not gap artifacts)
real_jumps = [d for d, t in zip(jump_dates, jump_types)
              if t == 'equipment_change']

# Correct the jumps
corrected, step_sizes = ts.correct_jumps(real_jumps)
```

#### Outlier Detection
```python
# Remove outliers using Median Absolute Deviation
cleaned = ts.detect_outliers(
    threshold=3.5,      # MAD z-score threshold (2-5 range)
    use_time=True       # Normalize rate by elapsed time
)
# Outliers are replaced with NaN
```

#### Gap Filling (SSA)
```python
# Fill missing values using Singular Spectrum Analysis
filled = ts.fill_gaps(
    method='ssa',
    embedding_dim=None,          # Auto-tuned if None
    n_components=None,           # Auto-selected if None
    variance_threshold=0.9,      # Retain 90% of variance
    max_components=None,
    max_iter=50,
    smooth_observed=False        # True = denoise all data
)
# Requirement: < 70% missing data
```

#### Smoothing
```python
# Centered moving average with NaN handling
smoothed = ts.smooth(window=7)
```

#### Interpolation
```python
# Spline interpolation (not a TimeSeries, returns arrays)
x_new, y_new = ts.interpolate_spline(
    factor=10,  # Output resolution multiplier
    k=3         # Spline degree (1=linear to 5=quintic)
)
```

#### Trend Analysis
```python
# Linear trend using RANSAC regression
trend, slope = ts.get_trend(
    method='linear',
    force_zero_intercept=False
)

# Polynomial trend
trend, coefficients = ts.get_trend(
    method='polynomial',
    order=2
)
```

#### Seasonality
```python
# Detect dominant seasonal patterns via FFT
seasonality_df = ts.find_seasonality()
# Returns DataFrame with columns: Amplitude, Frequency, Phase, Period (days)
# Sorted by amplitude descending
```

#### Sinusoidal Modeling
```python
# Fit sinusoidal model with multiple harmonic components
estimation, params = ts.fit_sinusoidal(
    periods=[365.25, 182.6]  # Periods in time steps (daily = days)
)
# params contains: baseline, trend_slope, and component details
```

#### Model Evaluation
```python
# Compute fit metrics
metrics = ts.evaluate_fit(trend_or_model)
# Returns: {'MSE': float, 'RMSE': float, 'MAE': float, 'R2': float}
```

#### Peaks & Troughs
```python
# Detect local extrema (requires no NaN)
peak_idx, trough_idx = ts.dropna().detect_peaks_troughs()

# Get peak-to-peak magnitudes
magnitudes = ts.find_peak_to_peak(peak_idx, trough_idx)
```

### Chaining Example

```python
# Full processing pipeline in one readable chain
result = (ts
    .align_to_fulltime(freq='D')
    .detect_outliers(threshold=3.5)
    .fill_gaps(method='ssa', variance_threshold=0.9)
    .smooth(window=15)
)

trend, slope = result.get_trend()
seasonality = result.find_seasonality()
```

---

## 2. TableArray

A tabular data container with Excel/CSV/JSON I/O and timeseries extraction.

### Import

```python
from appgeopy.core import TableArray
```

### Loading Data

```python
# From Excel
ta = TableArray.from_excel('data.xlsx', sheet_name='GPS')

# From CSV
ta = TableArray.from_csv('data.csv', parse_dates=['date'])

# From JSON
ta = TableArray.from_json('data.json')
```

### Saving Data

```python
# Append or create Excel sheet
ta.to_excel_sheet(
    'output.xlsx',
    sheet_name='results',
    mode='a',              # 'a' = append, 'w' = overwrite
    if_sheet_exists='replace',
    index=False
)

# Save to JSON
path = ta.to_json_file(
    folder_path='./output',
    file_name='data',
    indent=4
)
```

### Data Extraction

```python
# Extract a column as TimeSeries (automatic via __getitem__)
ts = ta['displacement']  # Returns TimeSeries, not Series!

# Extract with custom index
ts = ta.extract_timeseries(
    column='displacement',
    index_col='date'
)
```

### Data Manipulation

```python
# Align all dates to complete timeline
ta_full = ta.align_to_fulltime(freq='D')

# Convert cumulative measurements to incremental changes
incremental = ta.cumulative_to_incremental(['t1', 't2', 't3'])
```

### Static Helpers

```python
# List sheet names without loading entire file
sheet_names = TableArray.get_sheet_names('data.xlsx')

# JSON utilities
config = TableArray.read_json_to_dict('config.json')
TableArray.save_dict_to_json(config, './output', 'config')
```

---

## 3. SpatialTableArray

Geospatial data container with CRS transformations and spatial operations.

### Import

```python
from appgeopy.core import SpatialTableArray
```

### Loading Data

```python
# From various geospatial formats
spatial = SpatialTableArray.from_shapefile('stations.shp')
spatial = SpatialTableArray.from_geojson('data.geojson')
spatial = SpatialTableArray.from_geopackage('data.gpkg', layer='points')
spatial = SpatialTableArray.from_kml('survey.kml')

# From CSV with coordinates
spatial = SpatialTableArray.from_csv(
    'data.csv',
    x_col='lon',
    y_col='lat',
    crs='EPSG:4326'
)

# From DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3],
    'easting': [500000, 501000, 502000],
    'northing': [2000000, 2001000, 2002000]
})
spatial = SpatialTableArray.from_dataframe(
    df,
    x_col='easting',
    y_col='northing',
    crs='EPSG:32647'  # UTM zone 47N
)
```

### Saving Data

```python
spatial.to_shapefile('output.shp')
spatial.to_geojson('output.geojson')
spatial.to_geopackage('output.gpkg', layer='results')
spatial.to_kml('output.kml')
```

### Spatial Operations

```python
# Clip to polygon region
clipped = spatial.clip_to_polygon(region_gdf)

# Find nearby points
nearby = spatial.find_neighbors(
    target_gdf,
    key_column='station_id',
    buffer_radius=1000  # In CRS units
)

# Segment lines into fixed-length segments
segments = spatial.segment_lines(
    min_segment_length=100,
    line_name_field='LineName'
)
```

### Timeseries Extraction

```python
# Extract column as TimeSeries
ts = spatial.extract_timeseries(
    column='water_level',
    index_col='date'
)

# Generate point keys from coordinates
keys = spatial.generate_point_keys('lon', 'lat')
# Output: ['X100Y13', 'X101Y14', ...]
```

### GeoDataFrame Integration

Since SpatialTableArray subclasses `gpd.GeoDataFrame`, all standard geopandas methods work:

```python
# Access geometry
spatial.plot()
spatial.to_crs('EPSG:3857')  # Web Mercator

# Standard DataFrame operations
filtered = spatial[spatial['temperature'] > 20]
```

---

## 4. DataCube

Regular gridded spatial-temporal arrays with dimension-aware operations.

### Import

```python
from appgeopy.core import DataCube
```

### Construction

```python
import xarray as xr

# From NetCDF file
cube = DataCube.from_netcdf('temperature.nc')

# From DataFrame (will create regular grid)
df = pd.DataFrame({
    'date': dates,
    'x': x_coords,
    'y': y_coords,
    'temperature': temps
})
cube = DataCube.from_dataframe(
    df,
    time_col='date',
    x_col='x',
    y_col='y'
)

# From xarray.Dataset
ds = xr.Dataset({
    'temperature': (['time', 'x', 'y'], data),
    'time': times,
    'x': x_coords,
    'y': y_coords,
})
cube = DataCube(ds)
```

### Inspection

```python
# Display structure
cube.summarize()

# Query available times
times = cube.get_available_times()

# Extract variable as DataFrame
spatial_slice = cube.extract_spatial_slice(
    time='2023-06-15',
    var='temperature'
)
```

### Timeseries Operations

```python
# Extract 1D timeseries at point location
ts = cube.extract_timeseries(
    var='temperature',
    x=500000,
    y=2000000
)
# Returns: TimeSeries object

# Analyze at point
trend, slope = ts.get_trend()
seasonality = ts.find_seasonality()
```

### Spatial Operations

```python
# Temporal statistics
stats = cube.compute_temporal_statistics(
    variables='temperature',
    statistics=['mean', 'std', 'min', 'max']
)

# Temporal smoothing
smoothed = cube.smooth_temporal(window=7)
```

### Visualization

```python
# Plot spatial view at specific time
fig, ax = cube.plot_spatial_view('temperature', time_point='2023-06-15')

# Plot timeseries at point
fig, ax = cube.plot_timeseries('temperature', x=500000, y=2000000)
```

### Export

```python
# Save to NetCDF
cube.to_netcdf('output.nc')

# Save to Zarr
cube.to_zarr('output.zarr')
```

---

## Migration Guide

### From Legacy Functions to Core Classes

#### Jump Detection

**Old (function-based):**
```python
from appgeopy import detect_jumps, correct_jumps

jumps, types, gaps = detect_jumps(ts_series, penalty=25)
corrected, steps = correct_jumps(ts_series, jumps)
```

**New (class-based):**
```python
from appgeopy.core import TimeSeries

ts = TimeSeries(ts_series)
jumps, types, gaps = ts.detect_jumps(penalty=25)
corrected, steps = ts.correct_jumps(jumps)
```

#### Gap Filling

**Old:**
```python
from appgeopy.timeseriestools import impute_ssa

filled = impute_ssa(ts_series, variance_threshold=0.9)
```

**New:**
```python
ts = TimeSeries(ts_series)
filled = ts.fill_gaps(method='ssa', variance_threshold=0.9)
```

#### Trend Analysis

**Old:**
```python
from appgeopy import get_trend

trend, slope = get_trend(ts_series, method='linear')
```

**New:**
```python
ts = TimeSeries(ts_series)
trend, slope = ts.get_trend(method='linear')
```

---

## Key Differences from Legacy API

| Feature | Legacy | Core |
|---------|--------|------|
| **Interface** | Standalone functions | Methods on classes |
| **Chainability** | Not possible | Yes: `.method1().method2()` |
| **Type Safety** | Loose (pd.Series in/out) | Strict (TimeSeries preserved) |
| **Metadata** | Lost between operations | Preserved via `_metadata` |
| **Discoverability** | `help(function)` | `help(TimeSeries)`, IDE autocomplete |
| **Extensibility** | Difficult | Easy (subclass, override methods) |
| **Geospatial Support** | Limited (legacy geospatial.py) | Native (SpatialTableArray) |

---

## Common Workflows

### GPS/GNSS Displacement Analysis

```python
from appgeopy.core import TimeSeries

# Load GPS displacement data
ts = TimeSeries.from_csv('gps_data.csv', index_col='date', names=['displacement'])

# Full preprocessing pipeline
result = (ts
    .align_to_fulltime(freq='D')
    .detect_jumps(penalty=25)  # Detect equipment changes
    .correct_jumps([real_jumps])
    .detect_outliers(threshold=3.5)
    .fill_gaps(method='ssa')
    .smooth(window=15)
)

# Analyze trend
trend, slope = result.get_trend()
print(f"Displacement rate: {slope:.6f} m/day")

# Seasonal patterns
seasonality = result.find_seasonality()
print(seasonality.head())
```

### InSAR Time-Series

```python
from appgeopy.core import TimeSeries, DataCube

# Load InSAR displacement datacube (time × lat × lon)
cube = DataCube.from_netcdf('insar_deformation.nc')

# Extract timeseries at specific location
ts = cube.extract_timeseries(
    var='los_displacement',
    x=500000,
    y=2000000
)

# Decompose into trend + seasonal
trend, slope = ts.get_trend()
seasonality = ts.find_seasonality()
estimation, params = ts.fit_sinusoidal(periods=[365.25])
```

### Groundwater Level Monitoring

```python
from appgeopy.core import SpatialTableArray

# Load well network data
wells = SpatialTableArray.from_csv(
    'groundwater_stations.csv',
    x_col='lon',
    y_col='lat',
    crs='EPSG:4326'
)

# Extract timeseries for one well
ts = wells.extract_timeseries('water_level', index_col='date')

# Process and analyze
processed = (ts
    .detect_outliers(threshold=3.5)
    .fill_gaps(method='ssa')
    .smooth(window=7)
)

# Get depletion rate
trend, slope = processed.get_trend()
print(f"Depletion: {-slope:.6f} m/year")
```

### Multi-Station Climate Analysis

```python
from appgeopy.core import TableArray

# Load multi-station data
ta = TableArray.from_excel('climate_stations.xlsx', sheet_name='temperature')

# Process each station
for col in ta.columns:
    ts = ta[col]  # Automatic TimeSeries conversion

    filled = ts.fill_gaps(method='ssa')
    seasonality = filled.find_seasonality()

    print(f"{col}: {len(seasonality)} seasonal components")
```

---

## Best Practices

1. **Always align dates first:**
   ```python
   ts = ts.align_to_fulltime(freq='D')
   ```

2. **Use chaining for readability:**
   ```python
   result = ts.method1().method2().method3()
   ```

3. **Specify CRS explicitly for spatial data:**
   ```python
   spatial = SpatialTableArray.from_csv(..., crs='EPSG:4326')
   ```

4. **Check data quality before gap filling:**
   ```python
   n_missing = ts.isna().sum()
   if n_missing / len(ts) < 0.7:
       ts = ts.fill_gaps()
   ```

5. **Use `.dropna()` for methods requiring complete data:**
   ```python
   peaks, troughs = ts.dropna().detect_peaks_troughs()
   ```

6. **Preserve processing history:**
   ```python
   # Access metadata
   ts._processing_history  # List of operations applied
   ```

---

## Documentation

For detailed API documentation, use Python's built-in help:

```python
help(TimeSeries)
help(TimeSeries.detect_jumps)
help(SpatialTableArray.clip_to_polygon)
help(DataCube.extract_timeseries)
```

---

## Demo Programs

Three complete example programs are provided in `demo/`:

- **01_generate_dataset.py** — Create synthetic groundwater data
- **02_load_and_overview.py** — Explore data with SpatialTableArray
- **03_analysis_pipeline.py** — Full processing pipeline + visualization

See `demo/README.md` for usage instructions.

---

## Support

For issues, questions, or feature requests, refer to:
- CLAUDE.md — Project architecture & guidelines
- Core class docstrings — Detailed method documentation
- Demo programs — Practical examples
