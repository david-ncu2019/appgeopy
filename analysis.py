from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy
from numpy.fft import fft, fftfreq, ifft  # Fast Fourier Transform functions
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def find_peaks_troughs(data_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and troughs in a time-series data.

    Parameters:
    data_series (pd.Series): The time-series data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Indices of the peaks and troughs.
    """
    if not isinstance(data_series, pd.Series):
        raise ValueError("Input data_series must be a pandas Series.")

    if data_series.isnull().any():
        raise ValueError(
            "Input data_series contains NaN values. Please handle them before using this function."
        )

    peaks, _ = scipy.signal.find_peaks(data_series)
    troughs, _ = scipy.signal.find_peaks(-data_series)
    return peaks, troughs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def find_peak_to_peak(
    df: pd.DataFrame, peak_idx: List[int], trough_idx: List[int]
) -> pd.DataFrame:
    """
    Find the peak-to-peak values in the time-series data.

    Parameters:
    df (pd.DataFrame): The time-series data with datetime index.
    peak_idx (List[int]): Indices of the peaks.
    trough_idx (List[int]): Indices of the troughs.

    Returns:
    pd.DataFrame: DataFrame with peak and trough dates and values.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    if df.shape[1] != 1:
        raise ValueError("DataFrame must have a single column of data.")

    # if any(idx >= len(df) or idx < 0 for idx in peak_idx + trough_idx):
    #     raise IndexError(
    #         "Peak or trough indices are out of bounds of the DataFrame."
    #     )

    if df.isnull().any().any():
        raise ValueError(
            "Input DataFrame contains NaN values. Please handle them before using this function."
        )

    peak_dates = df.iloc[peak_idx].idxmax()[0]
    trough_dates = df.iloc[trough_idx].idxmin()[0]
    peak_values = df.iloc[peak_idx].max()[0]
    trough_values = df.iloc[trough_idx].min()[0]

    data_cache = {
        "date": [peak_dates, trough_dates],
        "value": [peak_values, trough_values],
    }

    result_df = pd.DataFrame(data_cache).set_index("date")
    result_df = result_df.sort_values(by="value", ascending=False)
    
    return result_df


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_linear_trend(series):
    """
    Returns the linear trend of a pandas series with missing values using RANSACRegressor.

    Parameters:
        series (pandas.Series): A pandas series with missing values and DatetimeIndex.

    Returns:
        pandas.Series: A pandas series representing the linear trend of the input series.
        float: The slope of the linear trend.
    """
    # Get x values
    x = np.arange(series.size)

    # Create mask for missing values
    isfinite = np.isfinite(series.values)

    # Fit RANSACRegressor to data
    X = x[isfinite].reshape(-1, 1)
    y = series.values[isfinite]
    linear_model = RANSACRegressor(random_state=42)
    linear_model.fit(X, y)

    # Get predicted values for linear model
    estimate = linear_model.predict(x.reshape(-1, 1))

    # Create pandas series from predicted values
    series_trend = pd.Series(estimate, index=series.index)

    return (series_trend, linear_model.estimator_.coef_[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_polynomial_trend(x, y, order, x_estimate=None):
    """
    Returns the polynomial trend of the given x and y arrays using RANSACRegressor.

    Parameters:
        x (array-like): The x-value array.
        y (array-like): The y-value array with potential missing values.
        order (int): The order of polynomial fitting.
        x_estimate (array-like, optional): The x-value array for estimating the y-value array. 
                                           If None, the input x-value array is used.

    Returns:
        pandas.Series: A pandas series representing the polynomial trend of the input series.
        array-like: The coefficients of the polynomial trend.
    """
    # Use input x-value array if x_estimate is not provided
    if x_estimate is None:
        x_estimate = x

    # Create mask for finite values
    is_finite = np.isfinite(y)

    # Prepare data for model fitting
    X = x[is_finite].reshape(-1, 1)
    y_finite = y[is_finite]

    # Create and fit the polynomial model using RANSAC
    polynomial_model = make_pipeline(PolynomialFeatures(order), RANSACRegressor(random_state=42))
    polynomial_model.fit(X, y_finite)

    # Predict values using the fitted model
    X_estimate = x_estimate.reshape(-1, 1)
    y_estimate = polynomial_model.predict(X_estimate)

    # Create a pandas series from the predicted values
    trend_series = pd.Series(y_estimate, index=x_estimate.flatten())

    # Get the coefficients of the polynomial trend
    coefficients = polynomial_model.named_steps['ransacregressor'].estimator_.coef_

    return trend_series, coefficients

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def find_seasonality(time_series_data, target_column=None, interval=1):
    """
    Analyze seasonality in a time series data using Fourier Transform.

    Parameters:
    time_series_data : pandas.DataFrame or pandas.Series
        DataFrame or Series containing the time series data.
    target_column : str, optional
        The name of the column to analyze if a DataFrame is provided. Not needed if a Series is provided.
    interval : int, optional
        The time interval between observations (default is 1).

    Returns:
    seasonal_summary : pandas.DataFrame
        A DataFrame containing amplitudes, frequencies, phases, and periods.
    """
    # Ensure the input data is a pandas DataFrame or Series
    if not isinstance(time_series_data, (pd.DataFrame, pd.Series)):
        raise ValueError("The input data must be a pandas DataFrame or Series.")

    # Ensure the index is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(time_series_data.index):
        raise ValueError("The index of the input data must be of datetime type.")

    # If input is a Series, convert it to a DataFrame
    if isinstance(time_series_data, pd.Series):
        time_series_data = time_series_data.to_frame(name='value')
        target_column = 'value'
    
    # Check if the target column exists in the DataFrame
    if target_column not in time_series_data.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")

    # Interpolating missing values in the target column
    signal = time_series_data[target_column].interpolate(method="linear")

    # Fourier Transform
    fourier_transform = fft(signal)
    n = len(signal) // 2  # Half the length for one-sided spectrum
    frequencies = fftfreq(len(fourier_transform), d=interval)[:n]

    # Extracting amplitudes and phases for the one-sided spectrum
    amplitudes = np.abs(fourier_transform)[:n] / n
    phases = np.angle(fourier_transform)[:n]
    periods_in_days = np.abs(1 / frequencies)

    # Creating a summary table
    summary_table = (
        pd.DataFrame(
            {
                "Amplitude": amplitudes,
                "Frequency": frequencies,
                "Phase": phases,
                "Period (days)": periods_in_days,
            }
        )
        .sort_values(by="Amplitude", ascending=False)
        .reset_index(drop=True)
    )

    return summary_table


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def correct_phase_shift(original_data, reconstructed_series, target_column=None):
    """
    Correct the phase shift between the original and reconstructed time series.

    Parameters:
    original_data : pandas.Series or pandas.DataFrame
        The original time series data. If DataFrame, `target_column` must be specified.
    reconstructed_series : numpy.ndarray
        The reconstructed signal obtained from the model.
    target_column : str, optional
        The name of the column in the original DataFrame representing the time series. Required if `original_data` is a DataFrame.

    Returns:
    numpy.ndarray
        The phase-corrected reconstructed signal.
    """
    # Ensure the input is a Series or DataFrame
    if isinstance(original_data, pd.DataFrame):
        if target_column is None:
            raise ValueError("target_column must be specified when original_data is a DataFrame.")
        if target_column not in original_data.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame.")
        original_series = original_data[target_column]
    elif isinstance(original_data, pd.Series):
        original_series = original_data
    else:
        raise TypeError("original_data must be either a pandas Series or DataFrame.")

    # Ensure that both signals are of the same length
    original_length = len(original_series)
    reconstructed_length = len(reconstructed_series)
    min_length = min(original_length, reconstructed_length)

    # Truncate both signals to the minimum length
    truncated_original_signal = original_series[:min_length].values
    truncated_reconstructed_signal = reconstructed_series[:min_length]

    # Handle missing values by using only finite values
    finite_mask = np.isfinite(truncated_original_signal)
    truncated_original_signal_finite = truncated_original_signal[finite_mask]
    truncated_reconstructed_signal_finite = truncated_reconstructed_signal[finite_mask]

    if len(truncated_original_signal_finite) == 0:
        raise ValueError("No finite values found in the original signal for phase correction.")

    # Remove the mean from each signal
    mean_adjusted_original_signal = truncated_original_signal_finite - np.nanmean(truncated_original_signal_finite)
    mean_adjusted_reconstructed_signal = truncated_reconstructed_signal_finite - np.nanmean(truncated_reconstructed_signal_finite)

    # Compute cross-correlation
    correlation = np.correlate(mean_adjusted_original_signal, mean_adjusted_reconstructed_signal, mode="full")

    # Find the index of maximum correlation
    max_corr_index = np.argmax(correlation)

    # Calculate the phase shift
    expected_max_corr_index = len(mean_adjusted_reconstructed_signal) - 1
    shift = max_corr_index - expected_max_corr_index

    # print(shift)

    # Correct the phase shift in the reconstructed signal
    phase_corrected_series = np.roll(truncated_reconstructed_signal, shift)

    return phase_corrected_series

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -