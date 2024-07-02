import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import List


def synthetic_daily_signal(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    linear_slope: float = 0.0,
    amplitude_list: List[float] = [1.0],
    period_list: List[float] = [1.0],
    variance: float = 0.01,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time-series data with multiple sinusoidal components,
    a linear trend, and random noise.

    Parameters:
    start_date (str): Start date of the time-series in 'YYYY-MM-DD' format.
    end_date (str): End date of the time-series in 'YYYY-MM-DD' format.
    linear_slope (float): Slope of the linear trend component.
    amplitude_list (List[float]): List of amplitudes for the sinusoidal components.
    period_list (List[float]): List of periods (in years) for the sinusoidal components.
    variance (float): Variance of the random noise component.
    random_seed (int): Seed for the random number generator.

    Returns:
    pd.DataFrame: DataFrame containing the generated time-series data.
    """
    np.random.seed(random_seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    days = (dates - dates[0]).days
    PI = np.pi

    # Create the seasonal component using numpy vectorization
    seasonal_component = np.sum(
        [
            amp * np.sin(2 * PI * days / (period * 365.25))
            for amp, period in zip(amplitude_list, period_list)
        ],
        axis=0
    )

    # Create a trend component
    trend_component = linear_slope * days

    # Create a random noise component
    noise_component = np.random.normal(scale=variance, size=len(dates))

    # Combine all components to create the time-series data
    data = seasonal_component + trend_component + noise_component

    return pd.DataFrame({"date": dates, "value": data}).set_index("date")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_sinusoidal_model_inputs(time_series_data, seasonality_info, select_col=None):
    """
    Prepare input parameters for the `fit_sinusoidal_model` function.

    This function filters out NaN values, generates numeric time arrays, 
    and extracts necessary parameters for sinusoidal modeling.

    Parameters:
    time_series_data : pandas.Series or pandas.DataFrame
        The time series data. If DataFrame, `select_col` must be specified. The index must be of datetime type.
    seasonality_info : pandas.DataFrame
        DataFrame containing the seasonality information with 'Amplitude', 'Periods', and 'Phase'.
    select_col : str, optional
        The column name in the DataFrame for which the sinusoidal model is to be fitted. Required if `time_series_data` is a DataFrame.

    Returns:
    tuple
        A tuple containing time values, observed values, amplitudes, periods, phase shifts, and baseline.
    """
    # Ensure the DataFrame index is datetime
    if not isinstance(time_series_data.index, pd.DatetimeIndex):
        raise ValueError("The index of the time series data must be of datetime type.")
    
    # Check if the input is a DataFrame or Series
    if isinstance(time_series_data, pd.DataFrame):
        if select_col is None:
            raise ValueError("select_col must be specified when time_series_data is a DataFrame.")
        if select_col not in time_series_data.columns:
            raise ValueError(f"Column '{select_col}' not found in DataFrame.")
        series_data = time_series_data[select_col]
    elif isinstance(time_series_data, pd.Series):
        series_data = time_series_data
    else:
        raise TypeError("time_series_data must be either a pandas Series or DataFrame.")
    
    # Ensure the seasonality_info DataFrame contains required columns
    required_columns = ["Amplitude", "Frequency", "Phase", "Period (days)"]
    if not all(col in seasonality_info.columns for col in required_columns):
        raise ValueError(f"seasonality_info must contain the following columns: {', '.join(required_columns)}")
    
    # Filter out NaN values
    notna_filter = series_data.notna()
    
    # Generate numeric time array for finite values
    numeric_time_arr = np.arange(len(series_data))
    numeric_time_arr_finite = numeric_time_arr[notna_filter]

    # Extract observed values for finite entries
    observed_values = series_data[notna_filter].values
    
    # Extract seasonality parameters
    amplitudes = seasonality_info["Amplitude"].values
    periods = seasonality_info["Period (days)"].values
    phase_shifts = seasonality_info["Phase"].values

    # Calculate the baseline as the mean of observed values
    baseline = np.nanmean(observed_values)

    return numeric_time_arr_finite, observed_values, amplitudes, periods, phase_shifts, baseline

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def sinusoidal_model(time_values, amplitude_terms, baseline):
    """
    Construct a sinusoidal model based on time, amplitude terms, and a baseline value.

    Parameters:
    time_values : array-like
        Array of time values.
    amplitude_terms : array-like
        Array of amplitude terms for each sinusoidal component.
    baseline : float
        Baseline value for the sinusoidal model.

    Returns:
    computed_values : array-like
        The values computed by the sinusoidal model.
    """
    return np.sum(amplitude_terms, axis=0) + baseline


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def least_squares_loss(parameters, time_values, observed_values, amplitudes, periods):
    """
    Compute the least squares loss for a sinusoidal model.

    Parameters:
    parameters : array-like
        Parameters for the sinusoidal model, including phase shifts and baseline.
    time_values : array-like
        Array of time values.
    observed_values : array-like
        Array of observed data values.
    amplitudes : array-like
        Amplitudes for each sinusoidal component.
    periods : array-like
        Periods for each sinusoidal component.

    Returns:
    residuals_squared : array-like
        Squared residuals of the observed data from the model.
    """
    num_seasons = len(amplitudes)
    amp_terms = np.zeros((num_seasons, len(time_values)))
    for i in range(num_seasons):
        amp_terms[i, :] = abs(amplitudes[i]) * np.sin(
            2 * np.pi * time_values / periods[i] + parameters[i]
        )
    residuals = observed_values - sinusoidal_model(time_values, amp_terms, parameters[-1])
    return np.square(residuals)  # np.square(residuals)  # Squaring the residuals


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def fit_sinusoidal_model(
    time_values, observed_values, amplitudes, periods, phase_shifts, baseline, predict_time=None
):
    """
    Fit a sinusoidal model to the observed data and estimate parameters.

    Parameters:
    time_values : array-like
        Time values for the fitting process.
    observed_values : array-like
        Observed data values.
    amplitudes : array-like
        Amplitude guesses for each sinusoidal component.
    periods : array-like
        Period guesses for each sinusoidal component.
    phase_shifts : array-like
        Phase shift guesses for each sinusoidal component.
    baseline : float
        Initial guess for the baseline of the model.
    predict_time : array-like, optional
        Time values for prediction using the fitted model.

    Returns:
    estimated_signal : array-like
        The signal estimated by the fitted sinusoidal model.
    """
    guess_params = np.concatenate((phase_shifts, [baseline]))
    tol = 1e-12

    try:
        run_lstsq = least_squares(
            least_squares_loss,
            guess_params,
            args=(time_values, observed_values, amplitudes, periods),
            loss="soft_l1",
            ftol=tol,
            xtol=tol,
            gtol=tol,
            method="trf",
            max_nfev=1000
            # tr_solver="lsmr",
        )
    except Exception as e:
        print(e)
        return None

    estimated_params = run_lstsq["x"]
    num_seasons = len(amplitudes)
    amp_terms = np.zeros((num_seasons, len(time_values)))
    for i in range(num_seasons):
        amp_terms[i, :] = abs(amplitudes[i]) * np.sin(
            2 * np.pi * time_values / periods[i] + estimated_params[i]
        )

    if predict_time is None:
        predict_time = time_values

    amp_terms2 = np.zeros((num_seasons, len(predict_time)))

    for i in range(num_seasons):
        amp_terms2[i, :] = abs(amplitudes[i]) * np.sin(
            2 * np.pi * predict_time / periods[i] + estimated_params[i]
        )

    estimation = sinusoidal_model(predict_time, amp_terms2, estimated_params[-1])

    return estimation


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -