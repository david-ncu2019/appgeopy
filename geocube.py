import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


class DataCube:
    """
    A unified class to create, inspect, visualize, and export a
    multi-dimensional data cube from either a DataFrame or a NetCDF file.
    """

    def __init__(self, xarray_dataset, temporal_coord=None, spatial_coords=None):
        """
        Initialize DataCube with optional coordinate specification.
        
        Parameters:
        - xarray_dataset: xarray Dataset
        - temporal_coord: str, name of temporal coordinate (auto-detect if None)
        - spatial_coords: list, names of spatial coordinates in desired order (auto-detect if None)
        """
        if not isinstance(xarray_dataset, xr.Dataset):
            raise TypeError("DataCube must be initialized with an xarray.Dataset.")
        
        self.data_cube = xarray_dataset
        
        # User-specified coordinates take priority
        if temporal_coord is not None:
            self.temporal_coord = temporal_coord
        else:
            self.temporal_coord = None
            
        if spatial_coords is not None:
            self.spatial_coords = spatial_coords
        else:
            self.spatial_coords = []
        
        # Auto-detect if not specified
        self._analyze_structure()

    @classmethod
    def from_dataframe(cls, df, time_col, x_col, y_col):
        """
        Creates a DataCube from a pandas DataFrame.
        """
        index_cols = [time_col, y_col, x_col]
        for col in index_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Error: Column '{col}' not found in the DataFrame."
                )

        df_indexed = df.set_index(index_cols)
        xarray_dataset = df_indexed.to_xarray()
        print("DataCube successfully created from DataFrame.")
        return cls(xarray_dataset)

    @classmethod
    def from_netcdf(cls, file_path):
        """
        Loads a DataCube from a NetCDF file without prior knowledge.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file was not found at: {file_path}")

        xarray_dataset = xr.open_dataset(file_path)
        print(
            f"DataCube successfully loaded from: {os.path.abspath(file_path)}"
        )
        return cls(xarray_dataset)

    def _analyze_structure(self):
        """Analyze structure, respecting user-specified coordinates."""
        
        # Auto-detect temporal coordinate if not specified
        if self.temporal_coord is None:
            for name, coord in self.data_cube.coords.items():
                if np.issubdtype(coord.dtype, np.datetime64):
                    self.temporal_coord = name
                    break
        
        # Auto-detect spatial coordinates if not specified
        if not self.spatial_coords:
            detected_spatial = []
            for name, coord in self.data_cube.coords.items():
                if coord.ndim == 1 and name != self.temporal_coord:
                    detected_spatial.append(name)
            
            # Sort for consistency
            self.spatial_coords = sorted(detected_spatial)
        
        # Validate specified coordinates exist
        missing_coords = [coord for coord in self.spatial_coords + [self.temporal_coord] 
                         if coord and coord not in self.data_cube.coords]
        if missing_coords:
            raise ValueError(f"Specified coordinates not found: {missing_coords}")

    def summarize(self):
        """Prints a comprehensive summary of the data cube's structure."""
        print("\n--- DataCube Summary ---")
        print(self.data_cube)
        print("\n" + "-" * 25)
        print("INFERRED STRUCTURE:")
        print(f"  - Temporal Coordinate: '{self.temporal_coord}'")
        print(f"  - Spatial Coordinates: {self.spatial_coords}")
        print(
            f"  - Data Variables (Features): {list(self.data_cube.data_vars)}"
        )
        print("--------------------------")

    def get_available_times(self):
        """
        Extracts and returns the list of time points directly from the data cube.

        Returns:
            list: A list of pandas Timestamp objects.
        """
        if not self.temporal_coord:
            print("No temporal coordinate was found in this DataCube.")
            return []
        # Convert xarray's time format to a list of pandas Timestamps
        return self.data_cube[self.temporal_coord].to_pandas().tolist()

    def export_to_netcdf(self, file_path):
        """Exports the data cube to a NetCDF file."""
        try:
            self.data_cube.to_netcdf(file_path)
            print(
                f"\nData cube successfully exported to: {os.path.abspath(file_path)}"
            )
        except Exception as e:
            print(f"An error occurred during export: {e}")

    def plot_spatial_view(self, feature, time_point=None, figsize=(8, 6), **kwargs):
        """Displays a 2D spatial heatmap for a specific feature at a given time."""
        if len(self.spatial_coords) < 2:
            print("Error: Need at least 2 spatial coordinates for spatial plot.")
            return None, None

        try:
            fig, ax = plt.subplots(figsize=figsize)

            # Check if this is a statistics cube (no temporal dimension)
            if (
                self.temporal_coord is None
                or self.temporal_coord not in self.data_cube[feature].dims
            ):
                # Statistics array - plot directly
                im = self.data_cube[feature].plot.pcolormesh(
                    x=self.spatial_coords[0], y=self.spatial_coords[1], 
                    ax=ax, **kwargs
                )
                ax.set_title(f"Spatial Map: {feature}")
            else:
                # Timeseries cube - need time selection
                if time_point is None:
                    print(f"Error: time_point required for timeseries data.")
                    print(f"Available times: {self.get_available_times()[:3]}...")
                    return None, None

                data_slice = self.data_cube.sel({self.temporal_coord: time_point})
                im = data_slice[feature].plot.pcolormesh(
                    x=self.spatial_coords[0], y=self.spatial_coords[1], 
                    ax=ax, **kwargs
                )
                ax.set_title(
                    f"Spatial View of '{feature}'\nat {pd.to_datetime(time_point).strftime('%Y-%m-%d')}"
                )

            ax.set_xlabel(f"{self.spatial_coords[0]}")
            ax.set_ylabel(f"{self.spatial_coords[1]}")
            ax.set_aspect("equal", adjustable="box")
            
            return fig, ax

        except KeyError:
            print(f"Error: Feature '{feature}' not found.")
            print(f"Available features: {list(self.data_cube.data_vars.keys())}")
            return None, None

    def plot_timeseries(self, feature, **coords):
        """Displays a time-series line plot for a feature at a specific location."""
        if not self.temporal_coord:
            print(
                "Error: Cannot plot time-series without a temporal coordinate."
            )
            return

        try:
            location_data = self.data_cube.sel(**coords)
            plt.figure(figsize=(12, 5))
            location_data[feature].plot.line(marker="o", ms=4)
            plt.title(f"Time-Series of '{feature}' at {coords}")
            plt.xlabel("Time")
            plt.ylabel(f"{feature}")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()
        except KeyError:
            print(f"Error: Location {coords} or feature '{feature}' not found.")

    def compute_temporal_statistics(
        self,
        variables: Union[str, List[str]] = None,
        statistics: Union[str, List[str]] = "mean",
    ) -> "DataCube":
        """
        Compute temporal statistics for specified variables.

        Parameters:
        - variables: Which data variables to process (None = all)
        - statistics: Which statistics to compute ('mean', 'std', 'min', 'max', etc.)

        Returns:
        - New DataCube with temporal statistics as spatial maps
        """
        if not self.temporal_coord:
            raise ValueError("No temporal coordinate found in DataCube.")

        # Handle variable selection
        if variables is None:
            variables = list(self.data_cube.data_vars.keys())
        elif isinstance(variables, str):
            variables = [variables]

        # Handle statistics selection
        if isinstance(statistics, str):
            statistics = [statistics]

        # Available statistics
        stat_functions = {
            "mean": lambda x, axis=0: np.nanmean(x, axis=axis),
            "std": lambda x, axis=0: np.nanstd(x, axis=axis),
            "min": lambda x, axis=0: np.nanmin(x, axis=axis),
            "max": lambda x, axis=0: np.nanmax(x, axis=axis),
            "median": lambda x, axis=0: np.nanmedian(x, axis=axis),
            "sum": lambda x, axis=0: np.nansum(x, axis=axis),
            "count": lambda x, axis=0: np.sum(~np.isnan(x), axis=axis),
        }

        # Create spatial-only coordinates
        new_coords = {
            coord: self.data_cube.coords[coord] for coord in self.spatial_coords
        }

        # Compute statistics
        computed_data = {}

        for var_name in variables:
            var_data = self.data_cube[var_name]
            data_array = var_data.values
            time_axis = var_data.dims.index(self.temporal_coord)

            for stat_name in statistics:
                if stat_name not in stat_functions:
                    print(
                        f"Warning: Unknown statistic '{stat_name}'. Skipping."
                    )
                    continue

                result = stat_functions[stat_name](data_array, axis=time_axis)
                result_var_name = f"{var_name}_{stat_name}"
                result_dims = [
                    dim for dim in var_data.dims if dim != self.temporal_coord
                ]
                computed_data[result_var_name] = (result_dims, result)

        # Create new dataset
        result_dataset = xr.Dataset(data_vars=computed_data, coords=new_coords)
        return DataCube(result_dataset)