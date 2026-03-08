"""
DataCube - A wrapper around xarray.Dataset for spatial-temporal data cubes.

Provides methods for creating, inspecting, visualizing, and exporting
multi-dimensional data cubes with temporal and spatial coordinates.

Supports extraction to TimeSeries and SpatialTableArray objects.

Examples
--------
>>> from appgeopy.core import DataCube
>>> cube = DataCube.from_netcdf('temperature_grid.nc')
>>> ts = cube.extract_timeseries(x=500000, y=2000000, var='temperature')
>>> help(DataCube)
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .timeseries import TimeSeries


class DataCube:
    """
    Multi-dimensional spatial-temporal data cube (wraps xarray.Dataset).

    Auto-detects temporal and spatial coordinates from the dataset.
    Supports creating cubes from DataFrames, NetCDF files, and Zarr stores.

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        The xarray Dataset to wrap.
    temporal_coord : str, optional
        Name of the temporal coordinate. Auto-detected if None.
    spatial_coords : list of str, optional
        Names of spatial coordinates. Auto-detected if None.

    Examples
    --------
    Create from NetCDF:

    >>> cube = DataCube.from_netcdf('data.nc')
    >>> cube.summarize()

    Create from DataFrame:

    >>> cube = DataCube.from_dataframe(df, time_col='date', x_col='easting', y_col='northing')

    Extract TimeSeries at a location:

    >>> ts = cube.extract_timeseries(x=500000, y=2000000, var='temperature')
    >>> trend, slope = ts.get_trend()
    """

    def __init__(self, xarray_dataset, temporal_coord=None, spatial_coords=None):
        if not isinstance(xarray_dataset, xr.Dataset):
            raise TypeError("DataCube must be initialized with an xarray.Dataset.")

        self.data = xarray_dataset
        self.temporal_coord = temporal_coord
        self.spatial_coords = spatial_coords if spatial_coords else []
        self._analyze_structure()

    # =========================================================================
    # Constructors
    # =========================================================================

    @classmethod
    def from_dataframe(cls, df, time_col, x_col, y_col):
        """
        Create a DataCube from a pandas DataFrame.

        The DataFrame must have columns for time, x-coordinate, and
        y-coordinate. All other columns become data variables.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame.
        time_col : str
            Column name for the time dimension.
        x_col : str
            Column name for the x spatial dimension.
        y_col : str
            Column name for the y spatial dimension.

        Returns
        -------
        DataCube

        Examples
        --------
        >>> cube = DataCube.from_dataframe(
        ...     df, time_col='date', x_col='easting', y_col='northing'
        ... )
        """
        index_cols = [time_col, y_col, x_col]
        for col in index_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        ds = df.set_index(index_cols).to_xarray()
        return cls(ds)

    @classmethod
    def from_netcdf(cls, filepath, **kwargs):
        """
        Load a DataCube from a NetCDF file.

        Parameters
        ----------
        filepath : str
            Path to the .nc file.
        **kwargs
            Additional arguments passed to xr.open_dataset().

        Returns
        -------
        DataCube

        Examples
        --------
        >>> cube = DataCube.from_netcdf('temperature.nc')
        >>> cube = DataCube.from_netcdf('large_file.nc', chunks={'time': 100})
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        ds = xr.open_dataset(filepath, **kwargs)
        return cls(ds)

    @classmethod
    def from_zarr(cls, store_path, **kwargs):
        """
        Load a DataCube from a Zarr store.

        Parameters
        ----------
        store_path : str
            Path to the Zarr store.
        **kwargs
            Additional arguments passed to xr.open_zarr().

        Returns
        -------
        DataCube

        Examples
        --------
        >>> cube = DataCube.from_zarr('data.zarr')
        """
        ds = xr.open_zarr(store_path, **kwargs)
        return cls(ds)

    # =========================================================================
    # Structure Analysis
    # =========================================================================

    def _analyze_structure(self):
        """Auto-detect temporal and spatial coordinates."""
        # Detect temporal coordinate
        if self.temporal_coord is None:
            for name, coord in self.data.coords.items():
                if np.issubdtype(coord.dtype, np.datetime64):
                    self.temporal_coord = name
                    break

        # Detect spatial coordinates
        if not self.spatial_coords:
            detected = []
            for name, coord in self.data.coords.items():
                if coord.ndim == 1 and name != self.temporal_coord:
                    detected.append(name)
            self.spatial_coords = sorted(detected)

        # Validate
        all_coords = self.spatial_coords + (
            [self.temporal_coord] if self.temporal_coord else []
        )
        missing = [c for c in all_coords if c and c not in self.data.coords]
        if missing:
            raise ValueError(f"Coordinates not found in dataset: {missing}")

    def summarize(self):
        """
        Print a comprehensive summary of the data cube structure.

        Shows the xarray Dataset representation, detected coordinates,
        and available data variables.

        Examples
        --------
        >>> cube.summarize()
        """
        print("\n--- DataCube Summary ---")
        print(self.data)
        print("\n" + "-" * 25)
        print("DETECTED STRUCTURE:")
        print(f"  Temporal Coordinate: '{self.temporal_coord}'")
        print(f"  Spatial Coordinates: {self.spatial_coords}")
        print(f"  Data Variables: {list(self.data.data_vars)}")
        print("-" * 25)

    def get_available_times(self):
        """
        Get the list of available time points.

        Returns
        -------
        list of pd.Timestamp
            Available time points, or empty list if no temporal coord.

        Examples
        --------
        >>> times = cube.get_available_times()
        >>> print(f"First: {times[0]}, Last: {times[-1]}")
        """
        if not self.temporal_coord:
            return []
        return self.data[self.temporal_coord].to_pandas().tolist()

    @property
    def variables(self):
        """List of data variable names in the cube."""
        return list(self.data.data_vars)

    # =========================================================================
    # Data Extraction
    # =========================================================================

    def extract_timeseries(self, var: Optional[str] = None, **coords: Any) -> TimeSeries:
        """
        Extract a 1D TimeSeries at a specific spatial location.

        Parameters
        ----------
        var : str, optional
            Data variable to extract. Required if multiple variables exist.
        **coords
            Coordinate values for spatial selection (e.g., x=500000, y=2000000).
            Uses nearest-neighbor selection.

        Returns
        -------
        TimeSeries

        Examples
        --------
        >>> ts = cube.extract_timeseries(var='temperature', x=500000, y=2000000)
        >>> trend, slope = ts.get_trend()
        """
        if var is None:
            vars_list = list(self.data.data_vars)
            if len(vars_list) == 1:
                var = vars_list[0]
            else:
                raise ValueError(
                    f"Multiple variables found: {vars_list}. "
                    "Specify which one with the 'var' parameter."
                )

        # Select location using nearest-neighbor
        selection = self.data.sel(method="nearest", **coords)
        series = selection[var].to_pandas()

        if isinstance(series, pd.Series):
            return TimeSeries(series, name=var)
        # If it collapsed to a scalar, return single-element TimeSeries
        return TimeSeries([series], name=var)

    def extract_spatial_slice(self, time: Union[str, pd.Timestamp], var: Optional[str] = None) -> pd.DataFrame:
        """
        Extract a 2D spatial slice at a specific time.

        Returns a pandas DataFrame with coordinate columns and the
        variable value. Can be converted to SpatialTableArray if needed.

        Parameters
        ----------
        time : str or pd.Timestamp
            Time point to extract.
        var : str, optional
            Data variable. Required if multiple exist.

        Returns
        -------
        pd.DataFrame
            DataFrame with spatial coordinates and values.

        Examples
        --------
        >>> spatial_df = cube.extract_spatial_slice(time='2023-06-15', var='temperature')
        """
        if var is None:
            vars_list = list(self.data.data_vars)
            if len(vars_list) == 1:
                var = vars_list[0]
            else:
                raise ValueError(
                    f"Multiple variables found: {vars_list}. Specify with 'var'."
                )

        sliced = self.data.sel({self.temporal_coord: time}, method="nearest")
        return sliced[var].to_dataframe().reset_index()

    # =========================================================================
    # Temporal Statistics
    # =========================================================================

    def compute_temporal_statistics(self, variables: Optional[Union[str, List[str]]] = None, statistics: Union[str, List[str]] = "mean") -> "DataCube":
        """
        Compute statistics across the time dimension.

        Produces spatial maps of mean, std, min, max, median, sum, or count.

        Parameters
        ----------
        variables : str or list of str, optional
            Variables to process. If None, processes all.
        statistics : str or list of str, optional
            Statistics to compute. Default is 'mean'.
            Options: 'mean', 'std', 'min', 'max', 'median', 'sum', 'count'.

        Returns
        -------
        DataCube
            New DataCube with computed statistics (no time dimension).

        Examples
        --------
        >>> stats = cube.compute_temporal_statistics(
        ...     variables='temperature',
        ...     statistics=['mean', 'std', 'max']
        ... )
        >>> stats.summarize()
        """
        if not self.temporal_coord:
            raise ValueError("No temporal coordinate found.")

        if variables is None:
            variables = list(self.data.data_vars)
        elif isinstance(variables, str):
            variables = [variables]

        if isinstance(statistics, str):
            statistics = [statistics]

        stat_funcs = {
            "mean": lambda x, axis: np.nanmean(x, axis=axis),
            "std": lambda x, axis: np.nanstd(x, axis=axis),
            "min": lambda x, axis: np.nanmin(x, axis=axis),
            "max": lambda x, axis: np.nanmax(x, axis=axis),
            "median": lambda x, axis: np.nanmedian(x, axis=axis),
            "sum": lambda x, axis: np.nansum(x, axis=axis),
            "count": lambda x, axis: np.sum(~np.isnan(x), axis=axis),
        }

        new_coords = {c: self.data.coords[c] for c in self.spatial_coords}
        computed = {}

        for var_name in variables:
            var_data = self.data[var_name]
            arr = var_data.values
            time_axis = var_data.dims.index(self.temporal_coord)

            for stat_name in statistics:
                if stat_name not in stat_funcs:
                    warnings.warn(f"Unknown statistic '{stat_name}'. Skipping.")
                    continue
                result = stat_funcs[stat_name](arr, axis=time_axis)
                result_dims = [
                    d for d in var_data.dims if d != self.temporal_coord
                ]
                computed[f"{var_name}_{stat_name}"] = (result_dims, result)

        ds = xr.Dataset(data_vars=computed, coords=new_coords)
        return DataCube(ds)

    # =========================================================================
    # Export
    # =========================================================================

    def to_netcdf(self, filepath):
        """
        Export the data cube to a NetCDF file.

        Parameters
        ----------
        filepath : str
            Output file path (.nc).

        Examples
        --------
        >>> cube.to_netcdf('output.nc')
        """
        self.data.to_netcdf(filepath)

    def to_zarr(self, store_path, **kwargs):
        """
        Export the data cube to a Zarr store.

        Parameters
        ----------
        store_path : str
            Output Zarr store path.
        **kwargs
            Additional arguments passed to Dataset.to_zarr().

        Examples
        --------
        >>> cube.to_zarr('output.zarr')
        """
        self.data.to_zarr(store_path, **kwargs)

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_spatial_view(self, feature, time_point=None, figsize=(8, 6),
                         **kwargs):
        """
        Display a 2D spatial heatmap for a feature at a given time.

        Parameters
        ----------
        feature : str
            Data variable name.
        time_point : str or pd.Timestamp, optional
            Time point to visualize. Required if data has a time dimension.
        figsize : tuple, optional
            Figure size. Default is (8, 6).
        **kwargs
            Additional arguments passed to xarray pcolormesh.

        Returns
        -------
        tuple of (fig, ax) or (None, None) on error.

        Examples
        --------
        >>> fig, ax = cube.plot_spatial_view('temperature', time_point='2023-06-15')
        """
        if len(self.spatial_coords) < 2:
            print("Error: Need at least 2 spatial coordinates.")
            return None, None

        try:
            fig, ax = plt.subplots(figsize=figsize)

            has_time = (
                self.temporal_coord is not None
                and self.temporal_coord in self.data[feature].dims
            )

            if not has_time:
                self.data[feature].plot.pcolormesh(
                    x=self.spatial_coords[0],
                    y=self.spatial_coords[1],
                    ax=ax,
                    **kwargs,
                )
                ax.set_title(f"Spatial Map: {feature}")
            else:
                if time_point is None:
                    print("Error: time_point required for time-series data.")
                    return None, None
                sliced = self.data.sel(
                    {self.temporal_coord: time_point}, method="nearest"
                )
                sliced[feature].plot.pcolormesh(
                    x=self.spatial_coords[0],
                    y=self.spatial_coords[1],
                    ax=ax,
                    **kwargs,
                )
                time_str = pd.to_datetime(time_point).strftime("%Y-%m-%d")
                ax.set_title(f"'{feature}' at {time_str}")

            ax.set_xlabel(self.spatial_coords[0])
            ax.set_ylabel(self.spatial_coords[1])
            ax.set_aspect("equal", adjustable="box")
            return fig, ax

        except KeyError:
            print(f"Error: Feature '{feature}' not found.")
            print(f"Available: {list(self.data.data_vars)}")
            return None, None

    def plot_timeseries(self, feature, figsize=(12, 5), **coords):
        """
        Display a time-series line plot at a specific location.

        Parameters
        ----------
        feature : str
            Data variable name.
        figsize : tuple, optional
            Figure size. Default is (12, 5).
        **coords
            Spatial coordinates for the location (e.g., x=500000, y=2000000).

        Returns
        -------
        tuple of (fig, ax)

        Examples
        --------
        >>> fig, ax = cube.plot_timeseries('displacement', x=500000, y=2000000)
        """
        if not self.temporal_coord:
            raise ValueError("No temporal coordinate for time-series plot.")

        location = self.data.sel(method="nearest", **coords)
        fig, ax = plt.subplots(figsize=figsize)
        location[feature].plot.line(marker="o", ms=4, ax=ax)
        ax.set_title(f"Time-Series of '{feature}' at {coords}")
        ax.set_xlabel("Time")
        ax.set_ylabel(feature)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig, ax

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __repr__(self):
        return f"DataCube(vars={self.variables}, temporal='{self.temporal_coord}', spatial={self.spatial_coords})"

    def __str__(self):
        return self.__repr__()
