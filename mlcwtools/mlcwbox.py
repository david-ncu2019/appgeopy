import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from appgeopy import *
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from your_package import h5pytools, datetime_handle, visualize, interpolate


class MLCW:
    """
    A flexible class to load MLCW data from HDF5, build DataFrames, and plot.
    Includes a single quick_plot method for both HDF5-based and custom DataFrames.
    """

    # ---------------------------------------------------------------------------------------------------------------
    def __init__(self, h5_fpath):
        """
        Initialize with the path to your HDF5 file.
        Actual data is loaded on demand, not at __init__ time, for flexibility.
        """
        self.h5_fpath = h5_fpath
        self.mlcw_data, self.mlcw_metadata = h5pytools.open_HDF5(self.h5_fpath)

    # ---------------------------------------------------------------------------------------------------------------
    def list_stations(self):
        """
        Returns a sorted list of station names in the HDF5 file.
        """
        return sorted(self.mlcw_data.keys())

    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def list_structure(d, prefix=""):
        keys = list(d.keys())  # Get all keys in the dictionary
        for i, key in enumerate(keys):
            connector = "├──" if i < len(keys) - 1 else "└──"  # Tree branching logic
            print(f"{prefix}{connector} {key}")  # Print the folder name

            value = d[key]
            if isinstance(value, dict):  # If it's a dictionary, go deeper
                new_prefix = prefix + ("│   " if i < len(keys) - 1 else "    ")  # Adjust indentation
                MLCW.list_structure(value, new_prefix)

    # ---------------------------------------------------------------------------------------------------------------
    def get_data(self):
        return self.mlcw_data, self.mlcw_metadata

    # ---------------------------------------------------------------------------------------------------------------
    def build_dataframe(
        self,
        station,
        value_type=None,
        start_date=None,
        end_date=None,
        reference_to_first_measurement=False,
    ):
        station_data = self.mlcw_data[station]
        array_by_valuetype = station_data["values"][value_type]
        date_arr = station_data["date"]
        depth_arr = station_data["depth"]

        # Convert byte-dates to Python datetime
        date_arr = [datetime_handle.bytes_to_datetime(ele) for ele in date_arr]

        # Build DataFrame
        try:
            if array_by_valuetype.shape[1] == len(date_arr):
                df = pd.DataFrame(data=array_by_valuetype, columns=date_arr, index=depth_arr)
            else:
                date_arr = date_arr[: array_by_valuetype.shape[1]]
                df = pd.DataFrame(data=array_by_valuetype, columns=date_arr, index=depth_arr)
        except:
            df = pd.DataFrame(data=array_by_valuetype, columns=date_arr)

        df = df.sort_index(axis="columns")

        # Filter by date range
        if start_date or end_date:
            df = df.loc[:, start_date:end_date]

        # Reference shift
        if reference_to_first_measurement and df.shape[1] > 0:
            df = df.subtract(df.iloc[:, 0], axis=0)

        return df

    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def plot_data_by_column(ax, df, borehole_depth):
        """
        Plot data from each column of a DataFrame against borehole depth,
        using smooth spline interpolation.
        """
        cmap = plt.get_cmap("turbo")
        norm = Normalize(vmin=df.columns.min().toordinal(), vmax=df.columns.max().toordinal())

        for select_col in df.columns:
            date_as_ordinal = select_col.toordinal()
            color = cmap(norm(date_as_ordinal))

            # Interpolate data
            select_array = df[select_col].values
            # x_new, y_new = interpolate.spline_interp(x=borehole_depth, y=select_array, factor=100)

            # Plot original data + spline
            ax.plot(-select_array, -borehole_depth, marker="o", linestyle="-", lw=1.5, color=color)
            # ax.plot(-y_new, -x_new, linestyle="-", lw=1.5, color=color)

    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def plotly_databyCol(df, borehole_depth, unit):
        """
        Plot data from each column of a DataFrame against borehole depth
        in an interactive Plotly figure (without smoothing).

        Parameters:
            df (pd.DataFrame): DataFrame where columns represent time-series data at different depths.
            borehole_depth (array-like): Corresponding depth values for each row of df.

        Returns:
            fig (plotly.graph_objects.Figure): Interactive figure with multiple lines.
        """

        fig = go.Figure()

        # Generate color scale from Turbo colormap
        num_colors = len(df.columns)
        colors = pc.sample_colorscale("Turbo", np.linspace(0, 1, num_colors))

        # Loop through each column (date) in the DataFrame
        for idx, select_col in enumerate(df.columns):
            select_array = df[select_col].values

            # Add scatter plot for original data (NO INTERPOLATION)
            fig.add_trace(
                go.Scatter(
                    x=-select_array,  # Negative for alignment
                    y=-borehole_depth,  # Negative to match original orientation
                    mode="markers+lines",
                    marker=dict(color=colors[idx], size=10, symbol="circle"),
                    line=dict(color=colors[idx], width=2),
                    name=str(select_col),  # Convert date to string
                )
            )

        # Customize layout
        fig.update_layout(
            title="Borehole Data Visualization",
            xaxis_title=f"Cumulative Compaction ({unit})",
            yaxis_title="Depth (m)",
            # yaxis=dict(autorange="reversed"),  # Ensure depth increases downward
            template="plotly_white",
            height=1200,
            width=1600,
        )

        return fig

    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def add_color_bar(fig, ax, df):
        """
        Add a vertical color bar inset to the figure, labeling with date strings.
        """
        modified_datetime = df.columns.strftime("%Y/%m/%d")
        sm = plt.cm.ScalarMappable(cmap="turbo", norm=Normalize(vmin=0, vmax=1))

        cbaxes = inset_axes(
            ax,
            width="25%",
            height="1.25%",
            loc="lower left",
            bbox_to_anchor=(0.7, 0.05, 0.1, 20),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sm, cax=cbaxes, orientation="vertical")
        cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        _len = len(modified_datetime)
        datestring_list = [
            modified_datetime[X] for X in [-1, int(_len * 0.8), int(_len * 0.6), int(_len * 0.4), int(_len * 0.2), 0]
        ]
        cbar.ax.set_yticklabels(datestring_list[::-1], fontsize=16)

    # ---------------------------------------------------------------------------------------------------------------
    def quick_plot(
        self,
        station=None,
        df=None,
        value_type=None,  # Added value_type for consistency with build_dataframe
        station_label=None,
        start_date=None,
        end_date=None,
        reference_to_first_measurement=False,
        unit="mm",
        figsize=(8.3 * 2 / 3, 11.7),
    ):
        """
        Quick plotting function for MLCW data.

        Parameters:
        - station (str, optional): Name of the station (used to retrieve data from HDF5).
        - df (pd.DataFrame, optional): Custom DataFrame (must be indexed by depth and have dates as columns).
        - value_type (str, optional): Type of value to extract from HDF5 (e.g., 'cleaned_ref2base', 'cleaned_ringbyring').
        - station_label (str, optional): Label for the plot title (defaults to station name or 'Custom Data').
        - start_date (str, optional): Start date for filtering data.
        - end_date (str, optional): End date for filtering data.
        - reference_to_first_measurement (bool, optional): If True, normalizes data to first measurement.
        - multiply_factor (float, optional): Scaling factor (default = 100 for cm conversion).
        - figsize (tuple, optional): Figure size for the plot.

        Returns:
        - None (Displays a plot)
        """

        # Check input consistency
        if station is not None and df is not None:
            raise ValueError("Provide either 'station' or 'df', not both.")
        elif station is None and df is None:
            raise ValueError("Must provide either 'station' or 'df' to plot.")

        # If user provides a station, build a DataFrame from HDF5
        if station is not None:
            if value_type is None:
                raise ValueError("Must specify 'value_type' when providing a station.")

            df = self.build_dataframe(
                station=station,
                value_type=value_type,  # New argument passed
                start_date=start_date,
                end_date=end_date,
                reference_to_first_measurement=reference_to_first_measurement,
            )
            if station_label is None:
                station_label = station
        else:
            # We have a custom df
            if station_label is None:
                station_label = "Custom Data"

        # Now df is guaranteed to be valid
        borehole_depth = df.index.values

        fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        self.plot_data_by_column(ax, df, borehole_depth)

        # Configure axis using your 'visualize' module
        visualize.configure_axis(
            ax,
            xlabel=f"Cumulative Compaction ({unit})",
            ylabel="Depth (m)",
            title=station_label,
            hide_spines=["bottom", "right"],
            major_tick_length=10,
            minor_tick_length=5,
            tick_direction="in",
            fontsize_base=14,
        )
        visualize.configure_ticks(
            ax=ax,
            # x_major_interval=50,
            # x_minor_interval=10,
            y_major_interval=50,
            y_minor_interval=10,
        )

        # Add color bar
        self.add_color_bar(fig, ax, df)

        # Flip x-axis to the top
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # Remove negative sign in y-axis tick labels
        current_ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
        new_ytick_labels = [txt.replace("−", "") for txt in current_ytick_labels]
        ax.set_yticklabels(new_ytick_labels)

        return fig, ax


# ---------------------------------------------------------------------------------------------------------------
    def quick_interactive(
        self,
        station=None,
        df=None,
        value_type=None,  # Added value_type for consistency with build_dataframe
        station_label=None,
        start_date=None,
        end_date=None,
        reference_to_first_measurement=False,
        unit="mm"
    ):
        """
        Quick plotting function for MLCW data using Plotly.

        Parameters:
        - station (str, optional): Name of the station (used to retrieve data from HDF5).
        - df (pd.DataFrame, optional): Custom DataFrame (must be indexed by depth and have dates as columns).
        - value_type (str, optional): Type of value to extract from HDF5 (e.g., 'cleaned_ref2base', 'cleaned_ringbyring').
        - station_label (str, optional): Label for the plot title (defaults to station name or 'Custom Data').
        - start_date (str, optional): Start date for filtering data.
        - end_date (str, optional): End date for filtering data.
        - reference_to_first_measurement (bool, optional): If True, normalizes data to first measurement.
        - multiply_factor (float, optional): Scaling factor (default = 100 for cm conversion).

        Returns:
        - fig (plotly.graph_objects.Figure): Interactive Plotly figure.
        """

        # Check input consistency
        if station is not None and df is not None:
            raise ValueError("Provide either 'station' or 'df', not both.")
        elif station is None and df is None:
            raise ValueError("Must provide either 'station' or 'df' to plot.")

        # If user provides a station, build a DataFrame from HDF5
        if station is not None:
            if value_type is None:
                raise ValueError("Must specify 'value_type' when providing a station.")

            df = self.build_dataframe(
                station=station,
                value_type=value_type,  # New argument passed
                start_date=start_date,
                end_date=end_date,
                reference_to_first_measurement=reference_to_first_measurement,
            )
            if station_label is None:
                station_label = station
        else:
            # We have a custom df
            if station_label is None:
                station_label = "Custom Data"

        # Now df is guaranteed to be valid
        borehole_depth = df.index.values

        # Use Plotly function instead of Matplotlib
        fig = self.plotly_databyCol(df, borehole_depth, unit)

        return fig