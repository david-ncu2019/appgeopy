"""
Robust Time Series Imputation & Smoothing via Iterative SSA

-----------------------------------------------------------------------------
This library provides a robust method for imputing and smoothing univariate
time series data. It is based on iterative Singular Spectrum Analysis (SSA),
which combines time-delay embedding with iterative low-rank reconstruction.

Goals:
- Fill gaps (missing values) in a consistent way.
- Extract the main components (trend + major oscillations).
- Avoid reconstructing small-scale, noisy variations.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


##########################################################################
# SECTION 1: CORE SSA IMPLEMENTATION
##########################################################################


def _create_embedding(time_series, embedding_dim, time_delay=1):
    """
    Creates the time-delay embedding (Hankel matrix) from a 1D series.

    Args:
        time_series (np.ndarray): The 1D time series.
        embedding_dim (int): The window size (L).
        time_delay (int): The lag (tau).

    Returns:
        embedded_matrix (np.ndarray): The (K x L) embedded matrix.
        start_indices (np.ndarray): The start indices for each row.
    """
    n = len(time_series)
    num_vectors = n - (embedding_dim - 1) * time_delay
    if num_vectors <= 0:
        raise ValueError(
            f"Embedding dimension too large. "
            f"Got n={n}, embedding_dim={embedding_dim}, time_delay={time_delay}."
        )

    embedded_matrix = np.full((num_vectors, embedding_dim), np.nan)

    for i in range(embedding_dim):
        start_idx = i * time_delay
        end_idx = start_idx + num_vectors
        embedded_matrix[:, i] = time_series[start_idx:end_idx]

    return embedded_matrix, np.arange(num_vectors)


def _reconstruct_original_series(embedded_matrix, original_length, embedding_dim):
    """
    Reconstructs the 1D series from the embedded matrix via diagonal averaging.

    Args:
        embedded_matrix (np.ndarray): The L-dimensional embedded matrix.
        original_length (int): The length (N) of the original 1D series.
        embedding_dim (int): The window size (L).

    Returns:
        np.ndarray: The reconstructed 1D time series.
    """
    reconstructed = np.full(original_length, np.nan)
    counts = np.zeros(original_length)

    num_vectors = embedded_matrix.shape[0]

    for i in range(num_vectors):
        for j in range(embedding_dim):
            pos = i + j
            if pos >= original_length:
                continue
            val = embedded_matrix[i, j]
            if not np.isnan(val):
                if np.isnan(reconstructed[pos]):
                    reconstructed[pos] = 0.0
                reconstructed[pos] += val
                counts[pos] += 1

    valid_indices = np.where(counts > 0)[0]
    reconstructed[valid_indices] /= counts[valid_indices]

    return reconstructed


def _select_components_by_variance(singular_values, threshold=0.9, max_components=None):
    """
    Select components that explain a cumulative variance threshold.

    Args:
        singular_values (np.ndarray): Singular values from SVD.
        threshold (float): Cumulative variance ratio to retain (0–1).
        max_components (int or None): Optional upper limit on components.

    Returns:
        int: Number of components to keep.
    """
    singular_values = np.asarray(singular_values)
    if singular_values.size == 0:
        return 1

    variance = singular_values ** 2
    variance_ratio = variance / variance.sum()
    cumulative_variance = np.cumsum(variance_ratio)

    # First index where cumulative variance >= threshold
    k = np.argmax(cumulative_variance >= threshold) + 1
    if k == 0:
        k = 1

    if max_components is not None:
        k = min(k, max_components)

    # Always keep at least 1, and not more than half of the available modes
    k = max(1, min(k, len(singular_values) // 2 if len(singular_values) > 2 else len(singular_values)))

    return k


def _iterative_ssa_solve(
    time_series,
    embedding_dim,
    n_components=None,
    variance_threshold=0.9,
    max_components=None,
    max_iter=30,
    tol=1e-5,
    smooth_observed=False,
):
    """
    Iterative SSA solver with variance-based component selection.

    This function:
    - Embeds the series,
    - Performs SVD,
    - Keeps only a low-rank approximation,
    - Reconstructs the 1D series,
    - Updates missing values iteratively.

    Args:
        time_series (np.ndarray): 1D series, detrended. May contain NaNs.
        embedding_dim (int): Window size (L).
        n_components (int or None): Fixed number of components (r).
            If None, it will be chosen by `variance_threshold`.
        variance_threshold (float): Cumulative variance to retain if
            `n_components` is None.
        max_components (int or None): Optional upper bound on r.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance (change in missing values).
        smooth_observed (bool): If True, smooth the entire series.
            If False, only fill missing values (preserve observed).

    Returns:
        np.ndarray: The 1D imputed and smoothed series (detrended).
    """
    current_series = np.array(time_series, dtype=float)
    missing_mask = np.isnan(current_series)

    # Initialize missing values with 0.0 (safe for detrended data)
    if np.all(missing_mask):
        # All values are missing; just return zeros
        return np.zeros_like(current_series)

    current_series[missing_mask] = 0.0
    prev_series = current_series.copy()

    for iteration in range(max_iter):
        X_embedded, _ = _create_embedding(current_series, embedding_dim, time_delay=1)
        X_embedded = X_embedded.astype(float)

        # Center columns
        X_mean = np.nanmean(X_embedded, axis=0)
        X_centered = X_embedded - X_mean

        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fall back to previous stable series
            return prev_series

        # Choose number of components
        if n_components is None:
            r = _select_components_by_variance(
                S,
                threshold=variance_threshold,
                max_components=max_components,
            )
        else:
            r = int(n_components)
            if r > len(S):
                r = len(S)
            # Do not let r be larger than embedding_dim - 1
            r = min(r, embedding_dim - 1 if embedding_dim > 1 else 1)

        # Low-rank reconstruction
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        X_recon = (U_r @ np.diag(S_r) @ Vt_r) + X_mean

        # Diagonal averaging back to 1D
        recon_1d = _reconstruct_original_series(
            X_recon, len(current_series), embedding_dim
        )

        # CRITICAL CHANGE: How to update the series
        if smooth_observed:
            # Replace ALL values with smoothed reconstruction
            diff = np.linalg.norm(recon_1d - prev_series)
            current_series[:] = recon_1d
        else:
            # Only replace missing values (preserve observed)
            diff = np.linalg.norm(recon_1d[missing_mask] - prev_series[missing_mask])
            current_series[missing_mask] = recon_1d[missing_mask]

        if diff < tol and iteration > 0:
            break

        prev_series = current_series.copy()

    return current_series


##########################################################################
# SECTION 2: PARAMETER SUGGESTION & DATA HANDLING
##########################################################################


def suggest_parameters(time_series):
    """
    Auto-tunes the embedding dimension (window size) based on the
    dominant (slowest) frequency in the data, found via FFT.

    Args:
        time_series (pd.Series or np.ndarray): The 1D time series.

    Returns:
        int: A suggested `embedding_dim` (window size).
    """
    y_temp = np.array(time_series, dtype=float).copy()
    n_total = len(y_temp)
    if n_total < 4:
        return max(2, n_total // 2)

    # Fill NaNs for FFT estimation
    if np.all(np.isnan(y_temp)):
        # All missing; just choose small window
        return max(2, n_total // 2)
    y_temp[np.isnan(y_temp)] = np.nanmean(y_temp)

    # Linear detrend for FFT
    x = np.arange(n_total)
    try:
        coeffs = np.polyfit(x, y_temp, 1)
        y_detrended = y_temp - np.polyval(coeffs, x)
    except np.linalg.LinAlgError:
        y_detrended = y_temp - np.mean(y_temp)

    # FFT
    yf = rfft(y_detrended)
    xf = rfftfreq(n_total, 1.0)

    # Ignore zero frequency
    if len(xf) <= 1:
        return max(2, n_total // 4)

    # Pick a few strongest frequencies (excluding zero)
    magnitudes = np.abs(yf[1:])
    if magnitudes.size == 0:
        return max(2, n_total // 4)

    peak_indices = np.argsort(magnitudes)[-3:]
    slowest_freq = np.min(xf[1:][peak_indices])

    if slowest_freq <= 1e-6:
        dominant_period = n_total / 4.0
    else:
        dominant_period = 1.0 / slowest_freq

    # Window: ~1.5x dominant period or 1/4 of length, but capped at N/2.1
    suggested = int(max(dominant_period * 1.5, n_total / 4.0))
    suggested = min(suggested, int(n_total / 2.1))
    suggested = max(2, min(suggested, n_total - 1))

    return suggested


def _handle_input_data(data, time_col=None, value_col=None):
    """
    Extract a clean 1D numpy array from various inputs.

    Args:
        data: pd.Series, pd.DataFrame, or np.ndarray
        time_col: (str, optional)
        value_col: (str, optional)

    Returns:
        (values, index):
            values (np.ndarray): 1D values (with NaNs)
            index (pd.Index or None): Original index if pandas, else None
    """
    if isinstance(data, pd.Series):
        return data.values.astype(float), data.index
    elif isinstance(data, pd.DataFrame):
        if value_col is None:
            raise ValueError("`value_col` must be provided for a DataFrame input.")
        df = data.copy()
        if time_col is not None and time_col in df.columns:
            df = df.sort_values(by=time_col)
            index = df[time_col]
        else:
            index = df.index
        return df[value_col].values.astype(float), index
    elif isinstance(data, np.ndarray):
        return data.flatten().astype(float), None
    else:
        raise TypeError("Input `data` must be a pd.Series, pd.DataFrame, or np.ndarray.")


def _reconstruct_output(imputed_values, original_index):
    """
    Format the output as a pd.Series if the input was a pandas object.
    """
    if original_index is not None:
        return pd.Series(imputed_values, index=original_index)
    return imputed_values


##########################################################################
# SECTION 3: PUBLIC-FACING API
##########################################################################


def impute_ssa(
    data,
    embedding_dim=None,
    n_components=None,
    variance_threshold=0.9,
    max_components=None,
    time_col=None,
    value_col=None,
    max_iter=50,
    tol=1e-5,
    smooth_observed=False,
):
    """
    Impute and smooth a time series using Iterative SSA.

    Design:
    - First, detrend the series (linear trend).
    - Then, apply SSA with low-rank reconstruction:
        * if n_components is given: use that fixed r.
        * else: choose r so that `variance_threshold` of energy is kept.
          You can also provide `max_components` to limit r.
    - Finally, add the trend back.

    Args:
        data (pd.Series, pd.DataFrame, np.ndarray): Time series.
        embedding_dim (int, optional): Window size (L). If None, auto-tuned.
        n_components (int or None): Fixed number of components.
        variance_threshold (float): Cumulative variance for component selection
            if `n_components` is None. Typical values: 0.8–0.95.
        max_components (int or None): Optional upper bound for r.
        time_col (str, optional): Time column name if `data` is a DataFrame.
        value_col (str, optional): Value column name if `data` is a DataFrame.
        max_iter (int): Max SSA iterations.
        tol (float): Convergence tolerance.
        smooth_observed (bool): If True, apply smoothing to ALL values (denoising).
            If False, only fill missing values (preserve observed points).

    Returns:
        pd.Series or np.ndarray: Imputed and smoothed time series.
    """
    # 1. Prepare data
    series_values, original_index = _handle_input_data(
        data, time_col=time_col, value_col=value_col
    )
    n = len(series_values)
    t = np.arange(n)

    # 2. Detrend (linear)
    valid_mask = ~np.isnan(series_values)
    if np.sum(valid_mask) < 2:
        mean_val = np.nanmean(series_values)
        if np.isnan(mean_val):
            mean_val = 0.0
        trend_line = np.full(n, mean_val)
    else:
        try:
            coeffs = np.polyfit(t[valid_mask], series_values[valid_mask], 1)
            trend_line = np.polyval(coeffs, t)
        except np.linalg.LinAlgError:
            trend_line = np.full(n, np.nanmean(series_values[valid_mask]))

    detrended_series = series_values - trend_line

    # 3. Auto-tune embedding dimension if needed
    if embedding_dim is None:
        embedding_dim = suggest_parameters(detrended_series)

    # Ensure embedding dimension is not too large
    embedding_dim = max(2, min(embedding_dim, n - 1))

    # 4. Iterative SSA solve (detrended)
    imputed_detrended = _iterative_ssa_solve(
        detrended_series,
        embedding_dim=embedding_dim,
        n_components=n_components,
        variance_threshold=variance_threshold,
        max_components=max_components,
        max_iter=max_iter,
        tol=tol,
        smooth_observed=smooth_observed,
    )

    # 5. Add trend back
    final_imputed_series = imputed_detrended + trend_line

    # 6. Reconstruct output format
    return _reconstruct_output(final_imputed_series, original_index)
