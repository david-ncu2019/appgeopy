import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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