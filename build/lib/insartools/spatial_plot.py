import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf, griddata

# Base size for A4 paper in inches (landscape orientation)
BASE_SIZE = (11.7, 8.27)

# ------------------------------------------------------------------------------


def calculate_scaling_factor(fig_width, fig_height, base_size=BASE_SIZE):
    """
    Calculate the scaling factor based on the figure size.

    Parameters:
    - fig_width (float): Width of the figure.
    - fig_height (float): Height of the figure.
    - base_size (tuple): Base size to scale from (width, height).

    Returns:
    - float: Scaling factor.
    """
    return (fig_width * fig_height) / (base_size[0] * base_size[1])


# ------------------------------------------------------------------------------


def configure_axis(
    ax, xlabel="", ylabel="", title="", scaling_factor=1, fontsize_base=18
):
    """
    Configure the axis labels, ticks, and title.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - title (str): Title of the axis.
    - scaling_factor (float): Scaling factor for font sizes.
    - fontsize_base (int): Base font size.
    """
    label_fontsize = fontsize_base * scaling_factor
    tick_label_fontsize = fontsize_base * scaling_factor * 1
    title_fontsize = fontsize_base * scaling_factor * 1.5

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ------------------------------------------------------------------------------


def configure_legend(ax, scaling_factor=1, fontsize_base=18):
    """
    Configure the legend.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - scaling_factor (float): Scaling factor for font sizes.
    - fontsize_base (int): Base font size.
    """
    legend_fontsize = fontsize_base * scaling_factor * 1
    ax.legend(fontsize=legend_fontsize, frameon=False)


# ------------------------------------------------------------------------------


def show_points(gdf, ax=None, color="blue", marker="o", alpha=0.7, **kwargs):
    """
    Plot point data from a GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing point data.
    - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new axes.
    - color (str): Color of the points.
    - marker (str): Marker style.
    - alpha (float): Transparency level.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - ax: Matplotlib Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    gdf.plot(ax=ax, color=color, marker=marker, alpha=alpha, **kwargs)

    return ax


# ------------------------------------------------------------------------------


def show_polygons(
    gdf, ax=None, facecolor="blue", edgecolor="black", alpha=0.5, **kwargs
):
    """
    Plot polygon data from a GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing polygon data.
    - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new axes.
    - facecolor (str): Color of the polygon faces.
    - edgecolor (str): Color of the polygon edges.
    - alpha (float): Transparency level.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - ax: Matplotlib Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    gdf.plot(
        ax=ax, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs
    )

    return ax


# ------------------------------------------------------------------------------


def add_basemap(ax, zoom=10, crs=None, source=ctx.providers.CartoDB.Positron):
    """
    Add a basemap to an existing plot.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes to add the basemap to.
    - zoom (int): Zoom level for the basemap tiles.
    - crs (str or dict, optional): Coordinate reference system of the basemap.
    - source (dict): Tile source provider from contextily. Defaults to Stamen TonerLite.

    Returns:
    - ax: Matplotlib Axes object with the basemap added.
    """
    if crs is None:
        crs = ax.get_xlim()[0].crs

    ctx.add_basemap(ax, zoom=zoom, crs=crs, source=source)
    return ax


# ------------------------------------------------------------------------------


def point_values(
    gdf,
    value_column,
    ax=None,
    cmap="viridis",
    marker="o",
    alpha=0.7,
    vmin=None,
    vmax=None,
    show_colorbar=True,
    **kwargs,
):
    """
    Plot points from a GeoDataFrame with values represented by a color bar.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing point data.
    - value_column (str): Column name in gdf to use for color mapping.
    - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new axes.
    - cmap (str or Colormap, optional): Colormap to use for the color bar.
    - marker (str): Marker style.
    - alpha (float): Transparency level.
    - vmin (float, optional): Minimum value for color normalization. If None, use min value from data.
    - vmax (float, optional): Maximum value for color normalization. If None, use max value from data.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - ax: Matplotlib Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Error handling for missing value_column
    if value_column not in gdf.columns:
        raise ValueError(
            f"'{value_column}' is not a column in the GeoDataFrame."
        )

    # Set default vmin and vmax if not provided
    if vmin is None:
        vmin = gdf[value_column].min()
    if vmax is None:
        vmax = gdf[value_column].max()

    # Normalize values for color mapping
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot points with a color map based on the specified column
    scatter = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf[value_column],
        cmap=cmap,
        norm=norm,
        marker=marker,
        alpha=alpha,
        **kwargs,
    )

    
    # Add color bar
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=50)
        # cbar.set_label(value_column)

    return ax


# ------------------------------------------------------------------------------


def show_anomalies(
    gdf,
    x_col,
    y_col,
    value_col,
    ax,
    threshold=10,
    label_offset=100,
    crs="EPSG:3826",
    name_col=None,
    name_color="blue",
    value_color="red",
):
    """
    Annotate significant points on a map based on values.

    This function annotates the names and values of stations on a map if their values
    exceed a specified threshold. The station name is shown below the point, and the
    value is shown above the point.

    Parameters:
    - gdf (GeoDataFrame or DataFrame): DataFrame containing the points to annotate.
    - x_col (str): Column name for the X coordinates.
    - y_col (str): Column name for the Y coordinates.
    - value_col (str): Column name for the values to check.
    - ax (matplotlib.axes.Axes): Axes to plot on.
    - threshold (float): Value threshold for significant annotation. Default is 10.
    - label_offset (float): Offset distance for labels from the point. Default is 100.
    - crs (str): Coordinate reference system of the points. Default is 'EPSG:3826'.
    - name_col (str, optional): Column name for the station names. If None, uses the DataFrame index.
    - name_color (str): Color for the station names. Default is 'blue'.
    - value_color (str): Color for the values. Default is 'red'.

    Returns:
    - None
    """
    try:
        # Convert DataFrame to GeoDataFrame if it's not already one
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=gpd.points_from_xy(gdf[x_col], gdf[y_col]),
                crs=crs,
            )
        else:
            # Ensure the GeoDataFrame has the correct CRS
            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)
            elif gdf.crs != crs:
                gdf.to_crs(crs, inplace=True)

        # Iterate over each row in the GeoDataFrame
        for idx, row in gdf.iterrows():
            try:
                # Extract coordinates and value
                x, y = row[x_col], row[y_col]
                value = row[value_col]
                name = row[name_col] if name_col else idx

                # Check if the value meets the specified threshold conditions
                if abs(value) > threshold:
                    # Add the station name below the point location
                    ax.text(
                        x,
                        y - label_offset,
                        name,
                        ha="center",
                        va="top",
                        fontweight="bold",
                        fontsize=10,
                        zorder=3,
                        color=name_color,
                    )

                    # Add the value above the point location
                    ax.text(
                        x,
                        y + label_offset,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        zorder=3,
                        color=value_color,
                    )

            except KeyError as e:
                print(f"KeyError: {e} for index: {idx}")
            except Exception as e:
                print(f"An error occurred for index {idx}: {e}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


# ------------------------------------------------------------------------------


def spatial_interpolation(
    data,
    x_col,
    y_col,
    value_col,
    method="linear",
    grid_resolution=100,
    cmap="viridis",
    colorbar_label="Interpolated Values",
    title="RBF Interpolation",
    xlabel="X",
    ylabel="Y",
    show_colorbar=True,
    alpha=0.7,
    figsize=(11.7, 8.27),
    vmin=None,
    vmax=None,
    ax=None,
    **kwargs,
):
    """
    Plot an interpolated spatial surface using Radial Basis Function (RBF) interpolation.

    This function creates a smooth, interpolated surface from scattered data points using RBF interpolation.
    It is useful for visualizing spatial variations and trends across a plane.

    Parameters:
    - data (pd.DataFrame): 
        DataFrame containing the data points with coordinates and values.
    - x_col (str): 
        Column name for the X coordinates in the DataFrame.
    - y_col (str): 
        Column name for the Y coordinates in the DataFrame.
    - value_col (str): 
        Column name for the values to interpolate in the DataFrame.
    - method (str, optional): 
        RBF function to use for interpolation. Options include:
        'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'.
        Default is 'linear'.
    - grid_resolution (int, optional): 
        Resolution of the grid for interpolation. Higher values produce finer grids.
        Default is 100.
    - cmap (str or Colormap, optional): 
        Colormap to use for the plot. Default is 'viridis'.
    - colorbar_label (str, optional): 
        Label for the colorbar. Default is 'Interpolated Values'.
    - title (str, optional): 
        Title of the plot. Default is 'RBF Interpolation'.
    - xlabel (str, optional): 
        Label for the X-axis. Default is 'X'.
    - ylabel (str, optional): 
        Label for the Y-axis. Default is 'Y'.
    - show_colorbar (bool, optional): 
        Whether to display the colorbar. Default is True.
    - alpha (float, optional): 
        Transparency level of the interpolated surface. Default is 0.7.
    - figsize (tuple, optional): 
        Size of the figure in inches (width, height). Default is (11.7, 8.27).
    - vmin (float, optional): 
        Minimum value for color normalization. If None, the minimum value from the data is used.
    - vmax (float, optional): 
        Maximum value for color normalization. If None, the maximum value from the data is used.
    - ax (matplotlib.axes.Axes, optional): 
        Axes to plot on. If None, new axes are created.
    - **kwargs: 
        Additional keyword arguments for RBF interpolation or contour plotting.

    Returns:
    - fig (matplotlib.figure.Figure): 
        The figure object containing the plot.
    - ax (matplotlib.axes.Axes): 
        The axes object containing the plot.

    Example usage:
    ```
    fig, ax = spatial_interpolation(
        data=df, 
        x_col='longitude', 
        y_col='latitude', 
        value_col='value', 
        method='linear', 
        grid_resolution=200, 
        cmap='plasma', 
        vmin=0, 
        vmax=100
    )
    plt.show()
    ```
    """
    # Extract coordinates and values from the DataFrame
    x = data[x_col].values
    y = data[y_col].values
    values = data[value_col].values

    # Create grid points for the interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_resolution),
        np.linspace(y.min(), y.max(), grid_resolution),
    )

    # Perform RBF interpolation
    try:
        rbf_interpolator = Rbf(x, y, values, function=method, **kwargs)
        grid_z = rbf_interpolator(grid_x, grid_y)
    except ValueError as e:
        raise ValueError(f"RBF interpolation failed: {e}")

    # If ax is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set default vmin and vmax if not provided
    if vmin is None:
        vmin = np.nanmin(grid_z)
    if vmax is None:
        vmax = np.nanmax(grid_z)

    # Normalize values for color mapping
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Define levels for contour to ensure it stays within vmin and vmax
    # levels = np.linspace(vmin, vmax, 100)
    levels = np.arange(vmin, vmax + 1, 5)

    # Plot the interpolated surface
    contour = ax.contourf(
        grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm, alpha=alpha
    )

    # Add a colorbar
    if show_colorbar:
        cbar = fig.colorbar(contour, ax=ax, pad=0.01, aspect=40)
        cbar.set_label(colorbar_label)

    # Configure the axes
    configure_axis(
        ax, xlabel=xlabel, ylabel=ylabel, title=title, fontsize_base=14
    )

    return fig, ax

# ------------------------------------------------------------------------------