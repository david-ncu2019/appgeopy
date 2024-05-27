import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_scaling_factor(fig_width, fig_height, base_size=(10, 6)):
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

def base_plot(ax, data, label, xlabel='', ylabel='', scaling_factor=1):
    """
    Base plotting function to handle common plotting features.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to plot on.
    - data (pd.Series): Data to plot.
    - label (str): Label for the data.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - scaling_factor (float): Scaling factor for font sizes.
    """
    base_fontsize = 10
    label_fontsize = base_fontsize * scaling_factor
    tick_label_fontsize = base_fontsize * scaling_factor * 0.8
    legend_fontsize = base_fontsize * scaling_factor * 0.9

    ax.plot(data, label=label)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)

def plot_single_graph(data, title='Single Graph', xlabel='X-axis', ylabel='Y-axis', figsize=(10, 6)):
    """
    Plot a single graph.

    Parameters:
    - data (pd.Series): Data to plot.
    - title (str): Title of the graph.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Size of the figure.
    """
    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    # Scale font sizes
    base_fontsize = 10
    title_fontsize = base_fontsize * scaling_factor * 1.5

    fig, ax = plt.subplots(figsize=figsize)
    base_plot(ax, data, label='Data', xlabel=xlabel, ylabel=ylabel, scaling_factor=scaling_factor)
    ax.set_title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.show()

def plot_three_subplots(df, figsize=(20, 12)):
    """
    Plot three subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data for each subplot.
    - figsize (tuple): Size of the figure.
    """
    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    # Scale font sizes
    base_fontsize = 10
    title_fontsize = base_fontsize * scaling_factor * 1.5
    label_fontsize = base_fontsize * scaling_factor

    fig = plt.figure(figsize=figsize)
    for i, col in enumerate(df.columns):
        ax = fig.add_subplot(3, 1, i + 1)
        select_array = df.loc[:, col]
        xlabel = 'Time' if i == len(df.columns) - 1 else ''
        base_plot(ax, select_array, label=col, xlabel=xlabel, ylabel=col, scaling_factor=scaling_factor)

    fig.suptitle('Three Subplots', fontsize=title_fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_custom_subplots(df, nrows, ncols, figsize=(20, 12), title='Custom Subplots'):
    """
    Plot custom subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data for each subplot.
    - nrows (int): Number of rows of subplots.
    - ncols (int): Number of columns of subplots.
    - figsize (tuple): Size of the figure.
    - title (str): Title of the figure.
    """
    fig_width, fig_height = figsize
    scaling_factor = calculate_scaling_factor(fig_width, fig_height)

    # Scale font sizes
    base_fontsize = 10
    title_fontsize = base_fontsize * scaling_factor * 1.5
    label_fontsize = base_fontsize * scaling_factor

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        select_array = df.loc[:, col]
        xlabel = 'Time' if i >= (nrows - 1) * ncols else ''
        base_plot(ax, select_array, label=col, xlabel=xlabel, ylabel=col, scaling_factor=scaling_factor)

    fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
