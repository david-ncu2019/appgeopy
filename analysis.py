import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from numpy.fft import fft, ifft, fftfreq  # Fast Fourier Transform functions


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


def find_seasonality(time_series_data, target_column, interval=1):
    """
    Analyze seasonality in a time series data using Fourier Transform.

    Parameters:
    time_series_data : pandas.DataFrame
        DataFrame containing the time series data.
    target_column : str
        The name of the column to analyze.
    interval : int, optional
        The time interval between observations (default is 1).

    Returns:
    seasonal_summary : pandas.DataFrame
        A DataFrame containing amplitudes, frequencies, phases, and periods.
    """

    # Check if the column exists in the DataFrame
    if target_column not in time_series_data.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")

    # Interpolating missing values
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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def correct_phase_shift(original_dataframe, reconstructed_series, target_column):
    """
    Correct the phase shift between the original and reconstructed time series.

    Parameters:
    original_dataframe : pandas.DataFrame
        DataFrame containing the original time series data.
    reconstructed_series : numpy.ndarray
        The reconstructed signal obtained from the model.
    target_column : str
        The name of the column in the original DataFrame representing the time series.

    Returns:
    phase_corrected_series : numpy.ndarray
        The phase-corrected reconstructed signal.
    """

    # Ensure that both signals are of the same length
    original_length = len(original_dataframe)
    reconstructed_length = len(reconstructed_series)
    min_length = min(original_length, reconstructed_length)

    # Truncate both signals to the minimum length
    truncated_original_signal = original_dataframe[target_column][:min_length].values
    truncated_reconstructed_signal = reconstructed_series[:min_length]

    # Removing the mean from each signal
    mean_adjusted_original_signal = truncated_original_signal - np.mean(truncated_original_signal)
    mean_adjusted_reconstructed_signal = truncated_reconstructed_signal - np.mean(
        truncated_reconstructed_signal
    )

    # Compute cross-correlation
    correlation = np.correlate(
        mean_adjusted_original_signal, mean_adjusted_reconstructed_signal, mode="full"
    )

    # Find the index of maximum correlation
    max_corr_index = np.argmax(correlation)

    # Calculate the phase shift
    expected_max_corr_index = len(mean_adjusted_reconstructed_signal) - 1
    shift = max_corr_index - expected_max_corr_index

    # Correct the phase shift in the reconstructed signal
    corrected_reconstructed_series = np.roll(truncated_reconstructed_signal, -shift)

    return corrected_reconstructed_series


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -