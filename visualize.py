import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# A4 paper size in inches (landscape)
BASE_SIZE = (11.7, 8.27)


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


def configure_axis(
    ax, xlabel="", ylabel="", title="", scaling_factor=1, fontsize_base=15
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


def configure_legend(ax, scaling_factor=1, fontsize_base=15):
    """
    Configure the legend.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to configure.
    - scaling_factor (float): Scaling factor for font sizes.
    - fontsize_base (int): Base font size.
    """
    legend_fontsize = fontsize_base * scaling_factor * 0.9
    ax.legend(fontsize=legend_fontsize, frameon=False)


def base_plot(
    ax,
    data,
    label,
    xlabel="",
    ylabel="",
    title="",
    scaling_factor=1,
    fontsize_base=15,
    **kwargs
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


def plot_single_graph(
    data,
    title="Single Graph",
    xlabel="X-axis",
    ylabel="Y-axis",
    figsize=(11.7, 8.27),
    **kwargs
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
        label="Data",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        scaling_factor=scaling_factor,
        **kwargs
    )
    fig.autofmt_xdate(rotation=90, ha="center")
    plt.tight_layout()
    return fig, ax


def plot_three_subplots(
    df,
    y_labels=None,
    x_labels=None,
    colors=None,
    figsize=(11.7, 8.27),
    suptitle="Three Subplots",
    **kwargs
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
            **kwargs
        )
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    fig.suptitle(suptitle, fontsize=18 * scaling_factor * 1.5, y=0.925, fontweight='bold')
    fig.autofmt_xdate(rotation=90, ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes

def save_figure(fig, savepath, dpi):
    fig.savefig(
        savepath,
        dpi=dpi,
        transparent=False,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

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