# appgeopy

A Python package for processing and analyzing geospatial time-series data.

Designed for environmental and geophysical workflows: GPS/GNSS displacement, InSAR time-series, groundwater monitoring, and regular gridded data cubes.

## Installation

```bash
git clone https://github.com/david-ncu2019/appgeopy.git
cd appgeopy
pip install -e .
```

For the full environment including geospatial extras:

```bash
pip install -e ".[full]"
```

> **Windows users:** Install `GDAL`, `fiona`, and `geopandas` via `conda` or OSGeo4W before running pip, as the binary wheels can conflict.

## Core Classes

| Class | Base | Purpose |
|---|---|---|
| `TimeSeries` | `pd.Series` | 1D time-indexed data (GPS, groundwater, InSAR, etc.) |
| `TableArray` | `pd.DataFrame` | Tabular data with Excel/CSV/JSON I/O |
| `SpatialTableArray` | `gpd.GeoDataFrame` | Geospatial tabular data (Shapefile, GeoJSON, GeoPackage, KML) |
| `DataCube` | `xr.Dataset` (wrapper) | Regular gridded spatial-temporal arrays (NetCDF, Zarr) |

## Quick Start

```python
from appgeopy import TimeSeries, TableArray, SpatialTableArray, DataCube
```

### TimeSeries

```python
import pandas as pd

# Create from data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
ts = TimeSeries(data, index=dates, name='displacement')

# Or generate synthetic data
ts = TimeSeries.synthetic(
    amplitude_list=[5.0, 2.0],
    period_list=[1.0, 0.5],
    linear_slope=0.001,
)

# Analysis pipeline
trend, slope = ts.get_trend(method='linear')
seasonality = ts.find_seasonality()
peaks, troughs = ts.dropna().detect_peaks_troughs()

# Preprocessing pipeline
jumps, types, gaps = ts.detect_jumps(penalty=25)
real_jumps = [d for d, t in zip(jumps, types) if t == 'equipment_change']
corrected, steps = ts.correct_jumps(real_jumps)
cleaned = corrected.detect_outliers(threshold=3.5)
filled = cleaned.fill_gaps(method='ssa', variance_threshold=0.9)
smoothed = filled.smooth(window=7)

# Sinusoidal modeling
estimation, params = ts.fit_sinusoidal(periods=[365.25, 182.6])

# Model evaluation
metrics = ts.evaluate_fit(trend)
```

### TableArray

```python
# Load data
ta = TableArray.from_excel('data.xlsx', sheet_name='GPS')
ta = TableArray.from_csv('data.csv', parse_dates=['date'])

# Extract time-series
ts = ta.extract_timeseries('displacement', index_col='date')

# Save
ta.to_excel_sheet('output.xlsx', 'results')

# Align to full timeline
ta_full = ta.align_to_fulltime(freq='D')

# Convert cumulative to incremental
incremental = ta.cumulative_to_incremental(date_columns)

# JSON operations
config = TableArray.read_json_to_dict('config.json')
TableArray.save_dict_to_json(config, './output', 'config')
```

### SpatialTableArray

```python
# Load from various formats
spatial = SpatialTableArray.from_shapefile('stations.shp')
spatial = SpatialTableArray.from_geojson('data.geojson')
spatial = SpatialTableArray.from_geopackage('data.gpkg', layer='points')
spatial = SpatialTableArray.from_kml('survey.kml')
spatial = SpatialTableArray.from_csv('data.csv', x_col='lon', y_col='lat', crs='EPSG:4326')

# Convert DataFrame to spatial
spatial = SpatialTableArray.from_dataframe(df, x_col='easting', y_col='northing', crs='EPSG:32647')

# Spatial operations
clipped = spatial.clip_to_polygon(region_gdf)
nearby = spatial.find_neighbors(target_gdf, 'station_id', buffer_radius=1000)
segments = lines.segment_lines(min_segment_length=100)

# Save to various formats
spatial.to_shapefile('output.shp')
spatial.to_geojson('output.geojson')
spatial.to_geopackage('output.gpkg', layer='results')
spatial.to_kml('output.kml')

# Extract time-series
ts = spatial.extract_timeseries('displacement', index_col='date')
```

### DataCube

```python
# Load
cube = DataCube.from_netcdf('temperature.nc')
cube = DataCube.from_dataframe(df, time_col='date', x_col='x', y_col='y')

# Inspect
cube.summarize()
times = cube.get_available_times()

# Extract TimeSeries at a point
ts = cube.extract_timeseries(var='temperature', x=500000, y=2000000)
trend, slope = ts.get_trend()

# Extract spatial slice
spatial_df = cube.extract_spatial_slice(time='2023-06-15', var='temperature')

# Compute statistics
stats = cube.compute_temporal_statistics(variables='temperature', statistics=['mean', 'std'])

# Visualize
fig, ax = cube.plot_spatial_view('temperature', time_point='2023-06-15')
fig, ax = cube.plot_timeseries('temperature', x=500000, y=2000000)

# Export
cube.to_netcdf('output.nc')
cube.to_zarr('output.zarr')
```

## Built-in Help

Every class and method has detailed docstrings with examples:

```python
help(TimeSeries)
help(TimeSeries.detect_jumps)
help(SpatialTableArray.clip_to_polygon)
help(DataCube.extract_timeseries)
```

## Tutorials

See the `tutorials/` folder for step-by-step notebooks.

## Dependencies

**Core:** numpy, pandas, matplotlib, scipy, seaborn, scikit-learn, xarray, ruptures, h5py, openpyxl, pyproj, shapely

**Geospatial extras (`pip install -e ".[geo]"`):** geopandas, fiona, GDAL

**Temporal modeling (`pip install -e ".[timeseries]"`):** prophet==1.1.1, holidays==0.24

## License

MIT
