from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression, Ridge


def synthetic_daily_signal(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    linear_slope: float = 0.0,
    amplitude_list: List[float] = [1.0],
    period_list: List[float] = [1.0],
    variance: float = 0.01,
    random_seed: int = 42,
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
        axis=0,
    )

    # Create a trend component
    trend_component = linear_slope * days

    # Create a random noise component
    noise_component = np.random.normal(scale=variance, size=len(dates))

    # Combine all components to create the time-series data
    data = seasonal_component + trend_component + noise_component

    return pd.DataFrame({"date": dates, "value": data}).set_index("date")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_sinusoidal_model_inputs(
    time_series_data, seasonality_info, select_col=None
):
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
        raise ValueError(
            "The index of the time series data must be of datetime type."
        )

    # Check if the input is a DataFrame or Series
    if isinstance(time_series_data, pd.DataFrame):
        if select_col is None:
            raise ValueError(
                "select_col must be specified when time_series_data is a DataFrame."
            )
        if select_col not in time_series_data.columns:
            raise ValueError(f"Column '{select_col}' not found in DataFrame.")
        series_data = time_series_data[select_col]
    elif isinstance(time_series_data, pd.Series):
        series_data = time_series_data
    else:
        raise TypeError(
            "time_series_data must be either a pandas Series or DataFrame."
        )

    # Ensure the seasonality_info DataFrame contains required columns
    required_columns = ["Amplitude", "Frequency", "Phase", "Period (days)"]
    if not all(col in seasonality_info.columns for col in required_columns):
        raise ValueError(
            f"seasonality_info must contain the following columns: {', '.join(required_columns)}"
        )

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

    return (
        numeric_time_arr_finite,
        observed_values,
        amplitudes,
        periods,
        phase_shifts,
        baseline,
    )


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


def least_squares_loss(
    parameters, time_values, observed_values, amplitudes, periods
):
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
    residuals = observed_values - sinusoidal_model(
        time_values, amp_terms, parameters[-1]
    )
    return np.square(
        residuals
    )  # np.square(residuals)  # Squaring the residuals


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def fit_sinusoidal_model(
    time_values,
    observed_values,
    amplitudes,
    periods,
    phase_shifts,
    baseline,
    predict_time=None,
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
    tol = 1e-10

    try:
        run_lstsq = least_squares(
            least_squares_loss,
            guess_params,
            args=(time_values, observed_values, amplitudes, periods),
            loss="soft_l1",  # "soft_l1" # "cauchy"
            ftol=tol,
            xtol=tol,
            gtol=tol,
            method="trf",
            jac="3-point",
            max_nfev=10_000,
            tr_solver="lsmr",
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

    estimation = sinusoidal_model(
        predict_time, amp_terms2, estimated_params[-1]
    )

    return estimation


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def fit_seasonal_model_with_uncertainty(
    time_values, observed_values, amplitudes, periods, phase_shifts, baseline
):
    """
    Enhanced seasonal fitting that returns parameter covariance for uncertainty estimation.

    Returns:
    --------
    estimation : np.ndarray
        Fitted seasonal predictions
    param_cov : np.ndarray
        Parameter covariance matrix
    fitted_params : np.ndarray
        Fitted parameters [phase_shifts..., baseline]
    """
    from scipy.optimize import least_squares

    guess_params = np.concatenate((phase_shifts, [baseline]))
    tol = 1e-10

    def residual_function(parameters):
        """Residual function for least squares optimization."""
        num_seasons = len(amplitudes)
        amp_terms = np.zeros((num_seasons, len(time_values)))
        for i in range(num_seasons):
            amp_terms[i, :] = abs(amplitudes[i]) * np.sin(
                2 * np.pi * time_values / periods[i] + parameters[i]
            )
        prediction = np.sum(amp_terms, axis=0) + parameters[-1]
        return observed_values - prediction

    try:
        # Fit using least_squares to get parameter covariance
        result = least_squares(
            residual_function,
            guess_params,
            loss="soft_l1",
            ftol=tol,
            xtol=tol,
            gtol=tol,
            method="trf",
            jac="3-point",
            max_nfev=10_000,
        )

        # Calculate parameter covariance matrix
        # For least_squares, we need to compute it from the Jacobian
        J = result.jac
        param_cov = np.linalg.inv(J.T @ J) * np.var(result.fun)

    except Exception as e:
        print(f"Enhanced seasonal fitting failed: {e}")
        return None, None, None

    # Generate predictions with fitted parameters
    estimated_params = result.x
    num_seasons = len(amplitudes)
    amp_terms = np.zeros((num_seasons, len(time_values)))
    for i in range(num_seasons):
        amp_terms[i, :] = abs(amplitudes[i]) * np.sin(
            2 * np.pi * time_values / periods[i] + estimated_params[i]
        )

    estimation = np.sum(amp_terms, axis=0) + estimated_params[-1]

    return estimation, param_cov, estimated_params


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def fit_sinusoidal_model_robust(
    time_values, 
    observed_values, 
    periods, 
    predict_time=None
):
    """
    Fits a sinusoidal model using Linear Regression on Fourier terms.
    This avoids the need for initial guesses on Amplitude or Phase.
    
    Parameters:
    -----------
    time_values : array-like (numeric, e.g. days)
    observed_values : array-like
    periods : list of floats (e.g. [365.25, 182.6])
    predict_time : array-like (optional)
    
    Returns:
    --------
    estimation : array-like (The fitted signal)
    model_params : dict (The extracted Amplitudes and Phases)
    """
    
    # 1. Construct the Feature Matrix (The Linearization Step)
    # For each period, we add a sin() and cos() column.
    # Model: y = baseline + Trend*t + Sum( C1*sin(wt) + C2*cos(wt) )
    
    X_train = time_values.reshape(-1, 1) # Start with Linear Trend
    
    feature_names = ['Trend']
    
    for p in periods:
        omega = 2 * np.pi / p
        sin_term = np.sin(omega * time_values)
        cos_term = np.cos(omega * time_values)
        
        # Stack new columns
        X_train = np.column_stack([X_train, sin_term, cos_term])
        feature_names.extend([f'Sin_{p:.1f}', f'Cos_{p:.1f}'])
        
    # 2. Fit the Model (Instant, Global Optimum)
    # We ignore NaNs during fitting
    valid_mask = ~np.isnan(observed_values)
    
    if np.sum(valid_mask) < len(feature_names) + 2:
        return None, {} # Not enough data
        
    reg = LinearRegression()
    reg.fit(X_train[valid_mask], observed_values[valid_mask])
    
    # 3. Predict
    if predict_time is None:
        predict_time = time_values
        X_pred = X_train
    else:
        # Reconstruct features for prediction time
        X_pred = predict_time.reshape(-1, 1)
        for p in periods:
            omega = 2 * np.pi / p
            sin_term = np.sin(omega * predict_time)
            cos_term = np.cos(omega * predict_time)
            X_pred = np.column_stack([X_pred, sin_term, cos_term])
            
    estimation = reg.predict(X_pred)
    
    # 4. Recover Physical Parameters (Amplitude & Phase)
    # Convert C1, C2 back to A, Phi
    # A = sqrt(C1^2 + C2^2)
    # Phi = atan2(C2, C1)
    
    coeffs = reg.coef_
    baseline = reg.intercept_
    trend_slope = coeffs[0]
    
    extracted_params = {
        'baseline': baseline,
        'trend_slope': trend_slope,
        'components': []
    }
    
    # coeffs[0] is Trend. Then pairs of (Sin, Cos) follow.
    for i, p in enumerate(periods):
        idx_sin = 1 + 2*i
        idx_cos = 1 + 2*i + 1
        
        c1 = coeffs[idx_sin]
        c2 = coeffs[idx_cos]
        
        amplitude = np.sqrt(c1**2 + c2**2)
        phase = np.arctan2(c2, c1)
        
        extracted_params['components'].append({
            'period': p,
            'amplitude': amplitude,
            'phase': phase
        })
        
    return estimation, extracted_params

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -