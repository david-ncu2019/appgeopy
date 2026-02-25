# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**appgeopy** is a geospatial and time-series analysis toolkit for processing environmental and geophysical data, with specialized support for:
- GPS/GNSS discontinuity detection and correction
- InSAR (Interferometric SAR) displacement analysis
- Groundwater and hydrological data processing
- Regular gridded data cubes (spatial-temporal)
- Irregular spatial data (shapefiles, point clouds)

**Current Status:** Architecture refactoring in progress (see "Architecture Evolution" below).

---

## Quick Start

### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install with all dependencies (from setup.py)
pip install numpy pandas geopandas matplotlib scipy seaborn scikit-learn prophet==1.1.1 holidays==0.24 xarray ruptures h5py
```

### Testing
No formal test suite exists yet. For now, test changes manually:
```bash
python -c "import appgeopy; print(appgeopy.__version__)"  # Verify import
```

---

## Project Structure

### Core Modules (root `appgeopy/`)

| Module | Purpose |
|--------|---------|
| `analysis.py` | Trend extraction (linear/polynomial), peaks/troughs, seasonality (FFT), phase correction, model evaluation metrics |
| `data_io.py` | Excel/JSON I/O, file backup utilities |
| `datetime_handle.py` | Date range alignment, time-series indexing, datetime parsing from filenames/bytes |
| `geospatial.py` | CRS transformation, point-in-polygon, spatial buffering, line segmentation (Shapely/GeoDataFrame) |
| `geocube.py` | `DataCube` class for regular 3D arrays (xarray wrapper) — handles netCDF, temporal-spatial grids |
| `interpolate.py` | B-spline interpolation (degree k=1–5) |
| `modeling.py` | Synthetic signal generation, sinusoidal fitting |
| `smoothing.py` | Centered moving average with NaN/edge handling |
| `visualize.py` | Matplotlib helpers (A4 sizing, axis formatting, DPI scaling) |

### Specialized Subpackages

| Package | Purpose |
|---------|---------|
| **`timeseriestools/`** | GPS/sensor discontinuity pipeline: jump detection (ruptures.Pelt), correction (anchor-based), gap imputation (SSA) |
| **`insartools/`** | InSAR time-series display and spatial-temporal visualization |
| **`gwatertools/`** | HDF5 utilities for groundwater/hydrological data |
| **`mlcwtools/`** | Machine learning & computational water modeling tools |

---

## Architecture Evolution

### Current State (as of Feb 2025)
- **Flat function structure:** Operations are standalone functions scattered across modules.
- **Limited composability:** Difficult to chain operations (e.g., `detect_jumps → correct_jumps → smooth`).
- **No unified interface:** Pandas Series, GeoDataFrames, xarray Datasets all handled via separate function calls.

### Planned Refactoring
The architecture will be unified into three core classes:

1. **`TimeSeries(pd.Series)`** — 1D time-indexed data with integrated methods:
   - `detect_jumps()`, `correct_jumps()`, `fill_gaps()` (imputation)
   - `get_trend()`, `detect_seasonality()`, `detect_peaks_troughs()`
   - `smooth()`, `interpolate_spline()`
   - Chainable: `ts.correct_jumps().smooth().fill_gaps()`

2. **`SpatialData(gpd.GeoDataFrame)`** — Irregular gridded spatial data:
   - `buffer_points()`, `clip_to_polygon()`, `segment_lines()`
   - `extract_timeseries(col)` → returns `TimeSeries`
   - CRS/geometry methods via GeoDataFrame inheritance

3. **`DataCube(xr.Dataset)`** — Regular gridded 3D arrays (time × x × y):
   - `extract_timeseries(x, y)` → `TimeSeries`
   - `extract_spatial_slice(time)` → `SpatialData`
   - `smooth_temporal()`, `plot_map()`, `plot_timeseries()`

**Operations reorganized into `ops/` package:**
- `ops/analysis.py`, `ops/preprocessing.py`, `ops/spatial.py`, `ops/datetime.py`, etc.
- Functions remain callable directly for advanced/custom workflows
- Classes call ops functions internally

### When Working on Refactoring
- **Do NOT break backward compatibility** until migration is complete.
- Keep existing module imports working (re-export from ops if needed).
- Write new methods on classes; keep old functions intact.
- Target: seamless transition where `appgeopy.detect_jumps()` and `ts.detect_jumps()` both work.

---

## Git Workflow

- **Main development branch:** `dev-new-feature` (refactoring happens here)
- **Production branch:** `main` (stable, published version)
- **Before pushing:** Ensure changes are on `dev-new-feature`, not `main`
- **Commit messages:** Concise, action-based (e.g., "Add DataCube.extract_timeseries()", "Fix jump detection gap filtering")

---

## Key Dependencies & Versions

| Dependency | Purpose | Version |
|------------|---------|---------|
| `pandas`, `numpy`, `scipy` | Core data science | — |
| `scikit-learn` | RANSAC trend fitting, LinearRegression | — |
| `geopandas`, `shapely`, `pyproj` | GIS operations, CRS transformation | — |
| `xarray` | Multi-dimensional data cubes | — |
| `ruptures` | Change-point detection (Pelt algorithm) | — |
| `matplotlib`, `seaborn` | Visualization | — |
| `prophet`, `holidays` | Temporal/holiday modeling | 1.1.1, 0.24 (pinned) |
| `h5py` | HDF5 file I/O | — |

---

## Common Patterns

### Time-Series Operations
Most operations expect `pd.Series` with DatetimeIndex. Ensure NaN handling:
```python
ts = pd.Series(values, index=dates)
# detect_jumps, correct_jumps, etc. expect this structure
```

### Geospatial Operations
Operations work with GeoDataFrames (from `gpd.GeoDataFrame`). CRS is critical:
```python
gdf = gpd.GeoDataFrame(geometry=geom, crs='EPSG:4326')
# Always specify CRS explicitly
```

### DataCube Usage
Expect xarray Datasets with named dimensions (e.g., `time`, `x`, `y`):
```python
ds = xr.Dataset({
    'temperature': (['time', 'x', 'y'], data),
    'time': times,
    'x': x_coords,
    'y': y_coords,
})
cube = DataCube(ds)
```

### Warnings Suppression
The main `__init__.py` suppresses all warnings globally. Be aware that errors may be silent. Override locally if debugging:
```python
import warnings
warnings.filterwarnings("default")  # Re-enable for debugging
```

---

## Refactoring Checklist (for future work)

- [ ] Create `core/base.py` with abstract base class (common methods)
- [ ] Implement `core/timeseries.py` with full method set
- [ ] Implement `core/spatialdata.py` with full method set
- [ ] Implement `core/datacube.py` (refine existing class)
- [ ] Reorganize functions into `ops/*.py` with re-exports for backward compatibility
- [ ] Add domain-specific subclasses: `InSARTimeSeries(TimeSeries)`, etc.
- [ ] Write integration tests for chained operations
- [ ] Update documentation with new API

---

## Notes for Contributors

- **Do not add tests to version control** unless they're part of the test suite (currently non-existent; create `tests/` when ready).
- **Keep operations stateless:** Functions should not depend on global state.
- **Type hints:** Use them for clarity, but not required for legacy code.
- **Docstrings:** Use numpy-style docstrings in new code.
- **Memory efficiency:** For large grids, use xarray for lazy evaluation where possible.

## Design Philosophy & Developer Workflow
* **Coding Style (Clean & Tidy):** Write clear, comprehensible, and straightforward Python/R code. Prioritize high readability and maintainability over clever, compact, or "pythonic" one-liners that obscure intent. 
* **Language & Documentation:** Write all docstrings, inline comments, and commit messages in simple, plain English (suitable for non-native speakers). Use short sentences and basic grammar. Explain technical jargon simply.
* **Modularity:** Strictly separate data loading, preprocessing, analysis, and visualization into distinct, reusable functions.
* **Parameterization:** Never hardcode values, file paths, or physical constants. Use parameterized variables or reference configuration files.
* **Data Integrity & Edge Cases:** Explicitly handle edge cases (e.g., missing dates, coordinate system mismatches, sensor noise). Do not trust default pandas/numpy/xarray behaviors for missing data. Validate intermediate outputs before proceeding.

## Token Efficiency & AI Interaction Rules
* **No Filler, High Density:** Respond in a concise, direct style. Avoid unnecessary repetition, boilerplate pleasantries, or extra explanations unless explicitly requested. Start directly with the core content.
* **Partial Outputs:** When modifying code, output ONLY the specific functions or classes that changed. Do not rewrite or print the entire file unless structurally necessary, to save tokens.
* **Ask Before Assuming:** If prompts or requirements are unclear, ambiguous, or missing crucial parameters, DO NOT make inferences. Stop and ask targeted clarifying questions to confirm understanding before generating code.
* **Act as an Intellectual Sparring Partner:** Do not simply agree with flawed logic. If an analytical approach seems geologically or mathematically unsound, point out the blind spots, question the assumptions, and present counterarguments or alternative interpretations using evidence.