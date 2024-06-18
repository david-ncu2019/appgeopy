import logging
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

# A4 paper size in inches (landscape)
BASE_SIZE = (11.7, 8.27)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def configure_legend(ax, scaling_factor=1, fontsize_base=18, frameon=False):
    """
    Configure the legend.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - scaling_factor (float): Scaling factor for font sizes.
    - fontsize_base (int): Base font size.
    """
    legend_fontsize = fontsize_base * scaling_factor * 1

    ax.legend(
        fontsize=legend_fontsize,
        frameon=frameon,
        labelspacing=0.1,
        handletextpad=0.2,
    )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def configure_ticks(
    ax,
    x_major_interval=None,
    x_minor_interval=None,
    y_major_interval=None,
    y_minor_interval=None,
):
    """
    Configure the major and minor tick intervals for the x and y axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - x_major_interval (float or None): Interval for major ticks on the x-axis.
    - x_minor_interval (float or None): Interval for minor ticks on the x-axis.
    - y_major_interval (float or None): Interval for major ticks on the y-axis.
    - y_minor_interval (float or None): Interval for minor ticks on the y-axis.

    Example Usage:
    ```
    fig, ax = plt.subplots()
    # Example data plotting
    ax.plot(range(100), range(100))

    # Configure ticks
    configure_ticks(ax, x_major_interval=10, x_minor_interval=5, y_major_interval=20, y_minor_interval=10)
    plt.show()
    ```
    """
    if x_major_interval is not None:
        x_major_loc = plticker.MultipleLocator(base=x_major_interval)
        ax.xaxis.set_major_locator(x_major_loc)

    if x_minor_interval is not None:
        x_minor_loc = plticker.MultipleLocator(base=x_minor_interval)
        ax.xaxis.set_minor_locator(x_minor_loc)

    if y_major_interval is not None:
        y_major_loc = plticker.MultipleLocator(base=y_major_interval)
        ax.yaxis.set_major_locator(y_major_loc)

    if y_minor_interval is not None:
        y_minor_loc = plticker.MultipleLocator(base=y_minor_interval)
        ax.yaxis.set_minor_locator(y_minor_loc)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def configure_datetime_ticks(
    ax,
    axis="x",
    major_interval=None,
    minor_interval=None,
    date_format="%Y/%m",
    start_date=None,
    end_date=None,
    grid=True,
):
    """
    Configure datetime tick labels and intervals for the specified axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - axis (str): Specify which axis to configure ('x' or 'y').
    - major_interval (int or None): Interval in months for major ticks. If None, no major ticks are set.
    - minor_interval (int or None): Interval in months for minor ticks. If None, no minor ticks are set.
    - date_format (str): Date format for the tick labels. Default is "%Y/%m".
    - start_date (datetime or None): Start date for the axis limit. If None, no limit is set.
    - end_date (datetime or None): End date for the axis limit. If None, no limit is set.
    - grid (bool): Whether to show grid lines for the major ticks. Default is True.

    Example Usage:
    ```
    fig, ax = plt.subplots()
    # Example data plotting
    dates = pd.date_range(start="2000-01-01", periods=100, freq="M")
    values = np.random.randn(100).cumsum()
    ax.plot(dates, values)

    # Configure datetime ticks
    configure_datetime_ticks(ax, axis="x", major_interval=24, minor_interval=12, date_format="%Y/%m",
                             start_date=datetime(2000, 1, 1), end_date=datetime(2023, 12, 31))
    plt.show()
    ```
    """
    if major_interval is not None:
        major_locator = mdates.MonthLocator(interval=major_interval)
        if axis == "x":
            ax.xaxis.set_major_locator(major_locator)
        elif axis == "y":
            ax.yaxis.set_major_locator(major_locator)
        else:
            raise ValueError("Axis must be 'x' or 'y'.")

    if minor_interval is not None:
        minor_locator = mdates.MonthLocator(interval=minor_interval)
        if axis == "x":
            ax.xaxis.set_minor_locator(minor_locator)
        elif axis == "y":
            ax.yaxis.set_minor_locator(minor_locator)
        else:
            raise ValueError("Axis must be 'x' or 'y'.")

    date_formatter = mdates.DateFormatter(date_format)
    if axis == "x":
        ax.xaxis.set_major_formatter(date_formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(date_formatter)

    if start_date and end_date:
        if axis == "x":
            ax.set_xlim(start_date, end_date)
        elif axis == "y":
            ax.set_ylim(start_date, end_date)

    if grid:
        ax.grid(which="major", axis=axis)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def base_plot(
    ax,
    data,
    label="",
    xlabel="",
    ylabel="",
    title="",
    scaling_factor=1,
    fontsize_base=18,
    **kwargs,
):
    """
    Base plotting function to handle common plotting features.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to plot on.
    - data (pd.Series): Data to plot.
    - label (str): Label for the data.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - title (str): Title for the plot.
    - scaling_factor (float): Scaling factor for font sizes.
    - fontsize_base (int): Base font size.
    - kwargs: Additional keyword arguments for the plot function.
    """
    ax.plot(data, label=label, **kwargs)
    configure_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        scaling_factor=scaling_factor,
        fontsize_base=fontsize_base,
    )
    configure_legend(
        ax, scaling_factor=scaling_factor, fontsize_base=fontsize_base
    )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_single_graph(
    data,
    title="Single Graph",
    xlabel="X-axis",
    ylabel="Y-axis",
    line_label="Data",
    figsize=(11.7, 8.27),
    **kwargs,
):
    """
    Plot a single graph.

    Parameters:
    - data (pd.Series): Data to plot.
    - title (str): Title of the graph.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Size of the figure.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.
    """
    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    base_plot(
        ax,
        data,
        label=line_label,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        scaling_factor=scaling_factor,
        **kwargs,
    )
    fig.autofmt_xdate(rotation=90, ha="center")
    plt.tight_layout()
    return fig, ax


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_three_subplots(
    df,
    y_labels=None,
    x_labels=None,
    colors=None,
    figsize=(11.7, 8.27),
    suptitle="Three Subplots",
    **kwargs,
):
    """
    Plot three subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data for each subplot.
    - y_labels (list): List of Y-axis labels for each subplot. If None, no Y-axis labels are set.
    - x_labels (list): List of X-axis labels for each subplot. If None, no X-axis labels are set.
                       If one element, label only the last subplot.
                       If three elements, label each subplot respectively.
    - colors (list): List of colors for each subplot.
    - figsize (tuple): Size of the figure.
    - suptitle (str): Super title for the figure.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - axes (numpy.ndarray): Array of axes objects.
    """
    if colors is None:
        colors = ["red", "green", "blue"]
    if y_labels is None:
        y_labels = [""] * len(df.columns)
    if x_labels is None:
        x_labels = [""] * len(df.columns)

    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    for i, (ax, col, color, ylabel) in enumerate(
        zip(axes, df.columns, colors, y_labels)
    ):
        select_array = df.loc[:, col]
        xlabel = ""
        if len(x_labels) == 1 and i == len(df.columns) - 1:
            xlabel = x_labels[0]
        elif len(x_labels) == len(df.columns):
            xlabel = x_labels[i]

        base_plot(
            ax,
            select_array,
            label=col,
            xlabel=xlabel,
            ylabel=ylabel,
            scaling_factor=scaling_factor,
            color=color,
            **kwargs,
        )
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    fig.suptitle(
        suptitle, fontsize=18 * scaling_factor * 1.5, y=0.925, fontweight="bold"
    )
    fig.autofmt_xdate(rotation=90, ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def save_figure(
    fig,
    savepath,
    dpi=300,
    transparent=False,
    facecolor="w",
    edgecolor="w",
    bbox_inches="tight",
):
    """
    Save a matplotlib figure to the specified file path with enhanced options.

    Parameters:
    - fig: matplotlib.figure.Figure
        The figure object to save.
    - savepath: str
        The full file path where the figure will be saved.
    - dpi: int, optional (default=300)
        The resolution of the saved figure in dots per inch.
    - transparent: bool, optional (default=False)
        Whether to make the figure background transparent.
    - facecolor: str, optional (default="w")
        The background color of the figure.
    - edgecolor: str, optional (default="w")
        The edge color of the figure.
    - bbox_inches: str or None, optional (default="tight")
        Bounding box in inches: 'tight' fits the figure tightly, None uses default padding.

    Returns:
    - None
    """

    # Validate file extension
    valid_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".svg",
        ".tiff",
        ".eps",
    ]
    file_extension = os.path.splitext(savepath)[-1].lower()
    if file_extension not in valid_extensions:
        raise ValueError(
            f"Invalid file extension: '{file_extension}'. Supported extensions are {valid_extensions}"
        )

    try:
        # Save the figure with specified parameters
        fig.savefig(
            savepath,
            dpi=dpi,
            transparent=transparent,
            facecolor=facecolor,
            edgecolor=edgecolor,
            bbox_inches=bbox_inches,
        )
        # logging.info(f"Figure saved successfully at '{savepath}'")
        # print(f"Figure saved successfully at '{savepath}'")
    except Exception as e:
        logging.error(f"Failed to save figure at '{savepath}': {e}")
        print(f"Failed to save figure at '{savepath}': {e}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_stem(
    data,
    title="Stem Plot",
    ax=None,
    xlabel="X-axis",
    ylabel="Y-axis",
    figsize=(11.7, 8.27),
    markerfmt="o",
    linefmt="-",
    basefmt=" ",
    **kwargs,
):
    """
    Plot a stem plot to highlight deviations or anomalies in a time series.

    Parameters:
    - data (pd.Series): Time-series data to plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Size of the figure, defaulting to A4 landscape (11.7 x 8.27 inches).
    - markerfmt (str): Format string for the markers.
                      Examples:
                      - 'o' for circle markers (default)
                      - '^' for triangle markers
                      - 's' for square markers
    - linefmt (str): Format string for the lines.
                    Examples:
                    - '-' for solid lines (default)
                    - '--' for dashed lines
                    - '-.' for dash-dot lines
    - basefmt (str): Format string for the baseline.
                    Examples:
                    - ' ' for no baseline (default)
                    - 'k-' for a solid black baseline
                    - 'r--' for a dashed red baseline
    - kwargs: Additional keyword arguments for customization, passed to `ax.stem`.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.

    Example usage:
    ```
    # Generate sample time-series data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()
    sample_data = pd.Series(values, index=dates)

    # Plot the stem plot with custom marker, line, and baseline formats
    fig, ax = plot_stem(
        sample_data,
        title="Custom Stem Plot",
        xlabel="Date",
        ylabel="Cumulative Sum",
        markerfmt='^',       # Triangle markers
        linefmt='r-.',       # Red dash-dot lines
        basefmt='k-'         # Solid black baseline
    )
    plt.show()
    """
    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.stem(
        data.index,
        data.values,
        markerfmt=markerfmt,
        linefmt=linefmt,
        basefmt=basefmt,
        **kwargs,
    )

    # configure_axis(
    #     ax,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     title=title,
    #     scaling_factor=scaling_factor,
    #     fontsize_base=18,
    # )

    # Customize grid for better visual clarity
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # fig.autofmt_xdate(rotation=90, ha="center")
    # fig.tight_layout()

    return ax


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# New horizontal bar plotting function
def plot_horizontal_bar(
    df,
    x_col,
    y_col,
    ax=None,
    title="Horizontal Bar Plot",
    xlabel="Value",
    ylabel="Category",
    color="skyblue",
    height=1,
    figsize=(11.7, 8.27),
):
    """
    Plot a horizontal bar plot for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to plot.
    - x_col (str): Column name for the x-axis values.
    - y_col (str): Column name for the y-axis labels.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - color (str): Color of the bars.
    - figsize (tuple): Size of the figure.
    - savepath (str or None): Path to save the figure. If None, the figure is not saved.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.
    """
    try:
        # Ensure the required columns are present in the DataFrame
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(
                f"Columns '{x_col}' or '{y_col}' not found in DataFrame"
            )

        fig_width, fig_height = figsize
        scaling_factor = calculate_scaling_factor(fig_width, fig_height)

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create the horizontal bar plot
        bars = ax.barh(df[y_col], df[x_col], color=color, height=height)

        # Annotate each bar with its value
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y()
                + bar.get_height() / 2,  # Position text at the end of each bar
                f"{width}",  # The text label
                va="center",  # Vertical alignment
                ha="left",  # Horizontal alignment
                fontsize=11 * scaling_factor,  # Scale the fontsize
            )

        # Configure the plot's appearance using the provided functions
        configure_axis(
            ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            scaling_factor=scaling_factor,
        )

        # If x_col is numeric, set the format
        if pd.api.types.is_numeric_dtype(df[x_col]):
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        # Tight layout for better spacing
        # fig.tight_layout()

        return ax

    except KeyError as ke:
        logging.error(f"KeyError: {ke}")
        print(f"KeyError: {ke}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


# def plot_custom_subplots(
#     df,
#     nrows,
#     ncols,
#     y_labels=None,
#     x_labels=None,
#     figsize=(11.7, 8.27),
#     title="Custom Subplots",
#     **kwargs
# ):
#     """
#     Plot custom subplots.

#     Parameters:
#     - df (pd.DataFrame): DataFrame containing the data for each subplot.
#     - nrows (int): Number of rows of subplots.
#     - ncols (int): Number of columns of subplots.
#     - y_labels (list): List of Y-axis labels for each subplot. If None, no Y-axis labels are set.
#     - x_labels (list): List of X-axis labels for each subplot. If None, no X-axis labels are set.
#     - figsize (tuple): Size of the figure.
#     - title (str): Title of the figure.
#     - kwargs: Additional keyword arguments for customization.

#     Returns:
#     - fig (matplotlib.figure.Figure): The figure object.
#     - axes (numpy.ndarray): Array of axes objects.
#     """
#     fig_width, fig_height = figsize
#     scaling_factor = calculate_scaling_factor(fig_width, fig_height)

#     fig, axes = plt.subplots(
#         nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height)
#     )
#     axes = axes.flatten()

#     if y_labels is None:
#         y_labels = [""] * len(df.columns)
#     if x_labels is None:
#         x_labels = [""] * len(df.columns)

#     for i, (ax, col, ylabel, xlabel) in enumerate(
#         zip(axes, df.columns, y_labels, x_labels)
#     ):
#         select_array = df.loc[:, col]
#         base_plot(
#             ax,
#             select_array,
#             label=col,
#             xlabel=xlabel,
#             ylabel=ylabel,
#             scaling_factor=scaling_factor,
#             **kwargs
#         )

#     fig.suptitle(title, fontsize=18 * scaling_factor * 1.5)
#     fig.autofmt_xdate(rotation=90, ha="center")
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     return fig, axes