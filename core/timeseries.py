"""
TimeSeries - A pandas Series subclass for time-indexed data analysis.

Extends pd.Series with methods for preprocessing, trend analysis,
seasonality detection, jump correction, gap imputation, smoothing,
and sinusoidal modeling.

Works with any 1D time-indexed data (GPS displacement, groundwater
levels, temperature, InSAR deformation, etc.).

Examples
--------
>>> from appgeopy.core import TimeSeries
>>> import pandas as pd
>>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
>>> ts = TimeSeries([...], index=dates, name='displacement')
>>> help(TimeSeries)
"""

from __future__ import annotations

import warnings
from itertools import islice
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class TimeSeries(pd.Series):
    """
    A time-indexed Series with integrated analysis, preprocessing, and modeling.

    Inherits all pandas.Series functionality and adds domain-specific methods
    for time-series analysis common in geosciences and environmental monitoring.

    Parameters
    ----------
    data : array-like, dict, or scalar
        Values for the time-series.
    index : pd.DatetimeIndex or datetime-like, optional
        Time index. Should be datetime for most methods to work.
    name : str, optional
        Name of the series.
    **kwargs
        Additional pandas.Series arguments.

    Examples
    --------
    Create from raw data:

    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> ts = TimeSeries(np.random.randn(100), index=dates, name='sensor_A')

    Create from existing pandas Series:

    >>> raw = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3))
    >>> ts = TimeSeries(raw)

    Chain operations:

    >>> result = ts.align_to_fulltime().smooth(window=7)
    """

    # Metadata attributes preserved during pandas operations
    _metadata = ["_processing_history"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_processing_history"):
            self._processing_history = []

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        # Avoids circular import; returns pd.DataFrame if TableArray not available
        return pd.DataFrame

    # =========================================================================
    # Time Alignment
    # =========================================================================

    def align_to_fulltime(self, freq: str = "D") -> "TimeSeries":
        """
        Extend the series to cover the full date range, filling gaps with NaN.

        Creates a complete timeline from the first to the last date in the
        index with the given frequency, inserting NaN where data is missing.

        Parameters
        ----------
        freq : str, optional
            Frequency string (e.g., 'D' for daily, 'H' for hourly).
            Default is 'D'.

        Returns
        -------
        TimeSeries
            Series aligned to the full time range.

        Examples
        --------
        >>> ts = TimeSeries([1, 2, 3], index=pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05']))
        >>> aligned = ts.align_to_fulltime(freq='D')
        >>> len(aligned)  # Now has 5 entries (Jan 1-5)
        5
        """
        if len(self) == 0:
            return self.copy()

        fulltime = pd.date_range(self.index[0], self.index[-1], freq=freq)
        missing_dates = fulltime.difference(self.index)

        if len(missing_dates) == 0:
            return self.copy()

        null_entries = pd.Series(
            np.nan, index=missing_dates, dtype=float, name=self.name
        )
        result = pd.concat([self, null_entries]).sort_index()
        result = TimeSeries(result)
        result._processing_history = self._processing_history + [
            f"align_to_fulltime(freq='{freq}')"
        ]
        return result

    def intersect_index(self, other_index):
        """
        Find common dates between this series and another index.

        Parameters
        ----------
        other_index : pd.Index or array-like
            The other time index to intersect with.

        Returns
        -------
        list
            Sorted list of common timestamps.

        Examples
        --------
        >>> common = ts1.intersect_index(ts2.index)
        """
        a = set(self.index)
        b = set(other_index)
        return sorted(list(a.intersection(b)))

    def numeric_index(self):
        """
        Generate numeric indices for non-null values.

        Useful for regression and modeling where numeric x-values are needed.

        Returns
        -------
        np.ndarray
            Array of integer indices where the series is not NaN.

        Examples
        --------
        >>> x = ts.numeric_index()
        >>> y = ts.dropna().values
        """
        non_null = self.notna()
        return np.arange(len(self))[non_null]

    # =========================================================================
    # Peaks & Troughs
    # =========================================================================

    def detect_peaks_troughs(self):
        """
        Find peaks and troughs in the time-series.

        The series must not contain NaN values. Use .dropna() or .fill_gaps()
        before calling this method if NaN values are present.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (peak_indices, trough_indices)

        Raises
        ------
        ValueError
            If the series contains NaN values.

        Examples
        --------
        >>> peaks, troughs = ts.dropna().detect_peaks_troughs()
        >>> print(f"Found {len(peaks)} peaks, {len(troughs)} troughs")
        """
        if self.isnull().any():
            raise ValueError(
                "Series contains NaN values. "
                "Use .dropna() or .fill_gaps() first."
            )
        peaks, _ = scipy.signal.find_peaks(self.values)
        troughs, _ = scipy.signal.find_peaks(-self.values)
        return peaks, troughs

    def find_peak_to_peak(self, peak_idx, trough_idx):
        """
        Extract peak and trough dates and values.

        Parameters
        ----------
        peak_idx : array-like
            Indices of detected peaks.
        trough_idx : array-like
            Indices of detected troughs.

        Returns
        -------
        pd.DataFrame
            DataFrame with all peak/trough dates, values, and type column.
            Columns: 'value', 'type' ('peak' or 'trough'). Sorted by value descending.

        Raises
        ------
        ValueError
            If the index is not DatetimeIndex or contains NaN.

        Examples
        --------
        >>> peaks, troughs = ts.detect_peaks_troughs()
        >>> summary = ts.find_peak_to_peak(peaks, troughs)
        >>> summary[summary['type'] == 'peak']
        """
        if not isinstance(self.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")
        if self.isnull().any():
            raise ValueError("Series contains NaN. Handle them first.")

        peak_df = self.iloc[peak_idx].to_frame(name="value")
        peak_df["type"] = "peak"
        trough_df = self.iloc[trough_idx].to_frame(name="value")
        trough_df["type"] = "trough"
        result = pd.concat([peak_df, trough_df]).sort_values("value", ascending=False)
        return result

    # =========================================================================
    # Trend Analysis
    # =========================================================================

    def get_trend(self, method: str = "linear", force_zero_intercept: bool = False,
                  order: int = 2, x_estimate: Optional[np.ndarray] = None) -> Tuple[Union["TimeSeries", pd.Series], Union[float, np.ndarray]]:
        """
        Fit a trend line to the series.

        Parameters
        ----------
        method : str, optional
            'linear' for linear trend (RANSAC), 'polynomial' for polynomial.
            Default is 'linear'.
        force_zero_intercept : bool, optional
            Force the linear intercept to zero. Only for method='linear'.
        order : int, optional
            Polynomial order. Only for method='polynomial'. Default is 2.
        x_estimate : array-like, optional
            Custom x-values for prediction (polynomial only).

        Returns
        -------
        tuple of (TimeSeries or pd.Series, float or np.ndarray)
            (trend_series, slope_or_coefficients)

        Examples
        --------
        >>> trend_line, slope = ts.get_trend(method='linear')
        >>> print(f"Slope: {slope:.4f} per time step")

        >>> trend_poly, coeffs = ts.get_trend(method='polynomial', order=3)
        """
        series = pd.to_numeric(self, errors="coerce")

        if method == "linear":
            x = np.arange(series.size)
            isfinite = np.isfinite(series.values).flatten()
            X = x[isfinite].reshape(-1, 1)
            y = series.values[isfinite]

            base_estimator = LinearRegression(
                fit_intercept=not force_zero_intercept
            )
            model = RANSACRegressor(estimator=base_estimator, random_state=42)
            model.fit(X, y)

            estimate = model.predict(x.reshape(-1, 1))
            trend = TimeSeries(estimate, index=series.index, name=self.name)
            slope = model.estimator_.coef_[0]
            return trend, slope

        elif method == "polynomial":
            x = np.arange(series.size)
            is_finite = np.isfinite(series.values)
            X = x[is_finite].reshape(-1, 1)
            y_finite = series.values[is_finite]

            if x_estimate is None:
                x_estimate = x

            try:
                poly_model = make_pipeline(
                    PolynomialFeatures(order),
                    RANSACRegressor(random_state=42),
                )
                poly_model.fit(X, y_finite)
                coefficients = (
                    poly_model.named_steps["ransacregressor"].estimator_.coef_
                )
            except (ValueError, np.linalg.LinAlgError):
                poly_model = make_pipeline(
                    PolynomialFeatures(order), LinearRegression()
                )
                poly_model.fit(X, y_finite)
                coefficients = poly_model.named_steps["linearregression"].coef_

            y_estimate = poly_model.predict(x_estimate.reshape(-1, 1))
            trend = pd.Series(y_estimate, index=x_estimate.flatten())
            return trend, coefficients

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'polynomial'.")

    # =========================================================================
    # Seasonality (FFT)
    # =========================================================================

    def find_seasonality(self, interval: float = 1) -> pd.DataFrame:
        """
        Analyze seasonality using Fourier Transform.

        Detects periodic patterns (frequency, amplitude, phase, period) in
        the time-series data.

        Parameters
        ----------
        interval : float, optional
            Sampling interval in days. If 1, auto-detected from the index.
            Default is 1.

        Returns
        -------
        pd.DataFrame
            Sorted by amplitude, with columns: Amplitude, Frequency,
            Phase, Period (days).

        Examples
        --------
        >>> seasonality = ts.find_seasonality()
        >>> dominant = seasonality.iloc[0]
        >>> print(f"Dominant period: {dominant['Period (days)']:.0f} days")
        """
        if not pd.api.types.is_datetime64_any_dtype(self.index):
            raise ValueError("Index must be datetime type for seasonality analysis.")

        # Auto-detect interval
        if interval == 1:
            time_diffs = self.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            std_dev = time_diffs.std()
            if std_dev > median_diff * 0.1:
                warnings.warn(
                    f"Irregular sampling detected (std/median={std_dev/median_diff:.2%}). "
                    "FFT assumes regular intervals. Results may be unreliable."
                )
            interval = median_diff.total_seconds() / (24 * 3600)

        signal = self.interpolate(method="linear")
        fourier_transform = fft(signal.values)
        n = len(signal) // 2
        frequencies = fftfreq(len(fourier_transform), d=interval)[:n]

        amplitudes = np.abs(fourier_transform)[:n] / n
        phases = np.angle(fourier_transform)[:n]
        periods = np.where(frequencies != 0, np.abs(1 / frequencies), np.inf)

        return (
            pd.DataFrame(
                {
                    "Amplitude": amplitudes,
                    "Frequency": frequencies,
                    "Phase": phases,
                    "Period (days)": periods,
                }
            )
            .sort_values(by="Amplitude", ascending=False)
            .reset_index(drop=True)
        )

    # =========================================================================
    # Phase Correction
    # =========================================================================

    def correct_phase_shift(self, reconstructed_series):
        """
        Correct phase shift between this series and a reconstructed signal.

        Uses cross-correlation to find and correct the optimal alignment.

        Parameters
        ----------
        reconstructed_series : np.ndarray
            The reconstructed signal to align.

        Returns
        -------
        np.ndarray
            Phase-corrected reconstructed signal.

        Examples
        --------
        >>> corrected = ts.correct_phase_shift(modeled_signal)
        """
        min_length = min(len(self), len(reconstructed_series))
        original = self.values[:min_length]
        reconstructed = reconstructed_series[:min_length]

        finite_mask = np.isfinite(original)
        orig_finite = original[finite_mask]
        recon_finite = reconstructed[finite_mask]

        if len(orig_finite) == 0:
            raise ValueError("No finite values found for phase correction.")

        orig_centered = orig_finite - np.nanmean(orig_finite)
        recon_centered = recon_finite - np.nanmean(recon_finite)

        correlation = np.correlate(orig_centered, recon_centered, mode="full")
        max_idx = np.argmax(correlation)
        shift = max_idx - (len(recon_centered) - 1)

        return np.roll(reconstructed, shift)

    # =========================================================================
    # Model Evaluation
    # =========================================================================

    def evaluate_fit(self, modeled_series):
        """
        Evaluate fit between this series and a modeled signal.

        Parameters
        ----------
        modeled_series : pd.Series or np.ndarray
            The modeled/predicted values.

        Returns
        -------
        dict
            Dictionary with MSE, RMSE, MAE, and R-squared.

        Examples
        --------
        >>> trend, slope = ts.get_trend()
        >>> metrics = ts.evaluate_fit(trend)
        >>> print(f"R2 = {metrics['R2']:.4f}")
        """
        if isinstance(modeled_series, np.ndarray):
            modeled_series = pd.Series(modeled_series, index=self.index)

        if len(self) != len(modeled_series):
            raise ValueError("Series lengths must match.")

        combined = pd.concat([self, modeled_series], axis=1).dropna()
        original = combined.iloc[:, 0]
        modeled = combined.iloc[:, 1]

        mse = np.mean((original - modeled) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - modeled))
        ss_res = np.sum((original - modeled) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    # =========================================================================
    # Smoothing
    # =========================================================================

    def smooth(self, window: int = 7) -> "TimeSeries":
        """
        Apply centered moving average smoothing.

        Handles NaN values by ignoring them in the window average.
        Pads edges with the first/last value to avoid shrinking.

        Parameters
        ----------
        window : int, optional
            Window size for the moving average. Default is 7.

        Returns
        -------
        TimeSeries
            Smoothed series with the same index.

        Examples
        --------
        >>> smoothed = ts.smooth(window=15)
        """
        half = window // 2
        first = self.iloc[0] if len(self) > 0 else np.nan
        last = self.iloc[-1] if len(self) > 0 else np.nan

        padded = [first] * half + list(self.values) + [last] * half

        def _sliding_window(iterable, size):
            it = iter(iterable)
            result = tuple(islice(it, size))
            if len(result) == size:
                yield result
            for elem in it:
                result = result[1:] + (elem,)
                yield result

        averages = [np.nanmean(w) for w in _sliding_window(padded, window)]

        result = TimeSeries(averages, index=self.index, name=self.name)
        result._processing_history = self._processing_history + [
            f"smooth(window={window})"
        ]
        return result

    # =========================================================================
    # Interpolation
    # =========================================================================

    def interpolate_spline(self, factor=10, k=3):
        """
        Interpolate using a B-spline curve.

        Parameters
        ----------
        factor : int, optional
            Multiplier for the number of output points. Default is 10.
        k : int, optional
            Spline degree (1=linear, 2=quadratic, 3=cubic, etc.).
            Must be between 1 and 5. Default is 3.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (x_new, y_new) - the interpolated coordinates.

        Examples
        --------
        >>> x_interp, y_interp = ts.interpolate_spline(factor=20, k=3)
        """
        if not (1 <= k <= 5):
            raise ValueError("k must be between 1 and 5.")

        x = np.arange(len(self))
        valid = self.notna()
        x_valid = x[valid]
        y_valid = self.values[valid]

        spline = make_interp_spline(x_valid, y_valid, k=k)
        x_new = np.linspace(x_valid.min(), x_valid.max(), len(x_valid) * factor)
        y_new = spline(x_new)
        return x_new, y_new

    # =========================================================================
    # Jump Detection & Correction (from timeseriestools)
    # =========================================================================

    def detect_data_gaps(self, gap_threshold_days: int = 30) -> List[Dict]:
        """
        Identify large data gaps between valid measurements.

        Parameters
        ----------
        gap_threshold_days : int, optional
            Minimum gap size in days to report. Default is 30.

        Returns
        -------
        list of dict
            Each dict has keys: 'start', 'end', 'days'.

        Examples
        --------
        >>> gaps = ts.detect_data_gaps(gap_threshold_days=60)
        >>> for g in gaps:
        ...     print(f"Gap: {g['start']} to {g['end']} ({g['days']} days)")
        """
        valid_indices = self.dropna().index
        gaps = []
        for i in range(len(valid_indices) - 1):
            gap_days = (valid_indices[i + 1] - valid_indices[i]).days
            if gap_days > gap_threshold_days:
                gaps.append(
                    {
                        "start": str(valid_indices[i].date()),
                        "end": str(valid_indices[i + 1].date()),
                        "days": int(gap_days),
                    }
                )
        return gaps

    def detect_jumps(self, penalty: int = 20, min_segment_days: int = 90, gap_threshold_days: int = 60) -> Tuple[List, List, List]:
        """
        Detect discontinuities using change-point detection (ruptures.Pelt).

        Automatically distinguishes between real jumps (equipment changes)
        and false positives caused by data gaps.

        Parameters
        ----------
        penalty : int, optional
            Ruptures penalty parameter. Higher = fewer jumps detected.
            Range: 15-30. Default is 20.
        min_segment_days : int, optional
            Minimum days between jumps. Default is 90.
        gap_threshold_days : int, optional
            Gap threshold for filtering. Default is 60.

        Returns
        -------
        tuple of (list, list, list)
            (jump_dates, jump_types, gaps)
            - jump_dates: list of pd.Timestamp
            - jump_types: list of str ('equipment_change' or 'gap_boundary')
            - gaps: list of gap dictionaries

        Examples
        --------
        >>> jump_dates, jump_types, gaps = ts.detect_jumps(penalty=25)
        >>> real_jumps = [d for d, t in zip(jump_dates, jump_types)
        ...              if t == 'equipment_change']
        >>> print(f"Found {len(real_jumps)} real jumps")
        """
        import ruptures as rpt

        valid_data = self.dropna()

        if len(valid_data) < 2 * min_segment_days:
            return [], [], []

        try:
            if len(valid_data) > 2000:
                # Downsample for large datasets
                factor = len(valid_data) // 2000
                down_vals = valid_data.iloc[::factor].values
                down_idx = valid_data.index[::factor]

                signal = down_vals.reshape(-1, 1)
                algo = rpt.Pelt(
                    model="rbf",
                    min_size=min_segment_days // factor,
                    jump=1,
                ).fit(signal)
                cps = algo.predict(pen=penalty)

                candidates = [down_idx[cp - 1] for cp in cps[:-1]]

                # Refine in +/-30 day window
                jump_dates_refined = []
                for approx in candidates:
                    w_start = approx - pd.Timedelta(days=30)
                    w_end = approx + pd.Timedelta(days=30)
                    window = valid_data[
                        (valid_data.index >= w_start) & (valid_data.index <= w_end)
                    ]
                    if len(window) > 20:
                        diffs = np.abs(np.diff(window.values))
                        refined = window.index[np.argmax(diffs)]
                        jump_dates_refined.append(refined)
                    else:
                        jump_dates_refined.append(approx)
            else:
                signal = valid_data.values.reshape(-1, 1)
                algo = rpt.Pelt(
                    model="rbf", min_size=min_segment_days, jump=1
                ).fit(signal)
                cps = algo.predict(pen=penalty)
                jump_dates_refined = [
                    valid_data.index[cp - 1] for cp in cps[:-1]
                ]

            # Filter out gap boundaries
            gaps = self.detect_data_gaps(gap_threshold_days=gap_threshold_days)
            jump_dates_filtered = []
            jump_types = []

            for date in jump_dates_refined:
                is_gap = False
                for gap in gaps:
                    gap_start = pd.to_datetime(gap["start"])
                    gap_end = pd.to_datetime(gap["end"])
                    if (
                        abs((date - gap_start).days) < 30
                        or abs((date - gap_end).days) < 30
                    ):
                        is_gap = True
                        break
                jump_dates_filtered.append(date)
                jump_types.append("gap_boundary" if is_gap else "equipment_change")

            return jump_dates_filtered, jump_types, gaps

        except (ValueError, RuntimeError) as e:
            warnings.warn(f"Jump detection error: {e}")
            return [], [], []

    def correct_jumps(self, jump_dates: List) -> Tuple["TimeSeries", List[float]]:
        """
        Correct jumps by aligning segments to the longest (anchor) segment.

        Uses median-based offset calculation with backward/forward passes
        from the anchor segment.

        Parameters
        ----------
        jump_dates : list of str or pd.Timestamp
            Dates where jumps occur (e.g., ['2022-06-17', '2023-01-10']).

        Returns
        -------
        tuple of (TimeSeries, list)
            (corrected_series, step_sizes)
            - corrected_series: Jump-corrected TimeSeries
            - step_sizes: list of step magnitudes at each jump

        Examples
        --------
        >>> jumps, types, gaps = ts.detect_jumps()
        >>> real_dates = [d for d, t in zip(jumps, types) if t == 'equipment_change']
        >>> corrected, steps = ts.correct_jumps(real_dates)
        """
        corrected = self.copy()

        if not jump_dates:
            return TimeSeries(corrected), []

        jump_dates_dt = sorted(set(pd.to_datetime(d) for d in jump_dates))

        # Find valid jump indices
        jump_indices = []
        for date in jump_dates_dt:
            if date in self.index:
                loc = self.index.get_loc(date)
                if isinstance(loc, slice):
                    loc = loc.start
                elif isinstance(loc, np.ndarray):
                    loc = np.where(loc)[0][0]
                if 0 < loc < len(self):
                    jump_indices.append(loc)

        if not jump_indices:
            return TimeSeries(corrected), []

        # Define segments
        segments = []
        start = 0
        for idx in jump_indices:
            segments.append((start, idx))
            start = idx
        segments.append((start, len(self)))

        # Find anchor (longest segment)
        lengths = [end - start for start, end in segments]
        anchor = np.argmax(lengths)

        offsets = np.zeros(len(segments))
        step_map = {}
        values = self.values

        # Backward pass (before anchor)
        for i in range(anchor - 1, -1, -1):
            curr_s, curr_e = segments[i]
            next_s, next_e = segments[i + 1]
            curr_vals = values[curr_s:curr_e]
            next_vals = values[next_s:next_e]
            curr_valid = curr_vals[~np.isnan(curr_vals)]
            next_valid = next_vals[~np.isnan(next_vals)]

            if len(curr_valid) >= 5 and len(next_valid) >= 5:
                n_edge = min(30, len(curr_valid) // 2, len(next_valid) // 2)
                target = np.median(next_valid[:n_edge] + offsets[i + 1])
                current = np.median(curr_valid[-n_edge:])
                offsets[i] = target - current
                step_map[next_s] = offsets[i + 1] - offsets[i]

        # Forward pass (after anchor)
        for i in range(anchor + 1, len(segments)):
            curr_s, curr_e = segments[i]
            prev_s, prev_e = segments[i - 1]
            curr_vals = values[curr_s:curr_e]
            prev_vals = values[prev_s:prev_e]
            curr_valid = curr_vals[~np.isnan(curr_vals)]
            prev_valid = prev_vals[~np.isnan(prev_vals)]

            if len(curr_valid) >= 5 and len(prev_valid) >= 5:
                n_edge = min(30, len(curr_valid) // 2, len(prev_valid) // 2)
                target = np.median(prev_valid[-n_edge:] + offsets[i - 1])
                current = np.median(curr_valid[:n_edge])
                offsets[i] = target - current
                step_map[curr_s] = offsets[i - 1] - offsets[i]

        # Apply offsets
        corrected_vals = corrected.values.copy().astype(float)
        for i, (s, e) in enumerate(segments):
            if offsets[i] != 0:
                corrected_vals[s:e] += offsets[i]

        result = TimeSeries(corrected_vals, index=self.index, name=self.name)
        result._processing_history = self._processing_history + [
            f"correct_jumps(n_jumps={len(jump_indices)})"
        ]

        final_steps = [step_map.get(idx, 0.0) for idx in jump_indices]
        return result, final_steps

    # =========================================================================
    # Outlier Detection
    # =========================================================================

    def detect_outliers(self, threshold: float = 3.5, use_time: bool = True) -> "TimeSeries":
        """
        Detect outliers using rate-of-change and Modified Z-Score (MAD).

        Parameters
        ----------
        threshold : float, optional
            MAD threshold. Higher = fewer outliers removed.
            Typical range: 2.0 to 5.0. Default is 3.5.
        use_time : bool, optional
            Normalize rates by elapsed time. Default is True.

        Returns
        -------
        TimeSeries
            Copy with outliers replaced by NaN.

        Examples
        --------
        >>> cleaned = ts.detect_outliers(threshold=3.0)
        >>> n_removed = cleaned.isna().sum() - ts.isna().sum()
        >>> print(f"Removed {n_removed} outliers")
        """
        original_nan = self.isna()
        valid_indices = np.where(~original_nan)[0]

        if len(valid_indices) < 3:
            return self.copy()

        rates = pd.Series(index=self.index, dtype=float)

        for i in range(1, len(valid_indices)):
            curr = valid_indices[i]
            prev = valid_indices[i - 1]
            change = self.iloc[curr] - self.iloc[prev]

            if use_time and isinstance(self.index, pd.DatetimeIndex):
                elapsed = (self.index[curr] - self.index[prev]).total_seconds() / 86400
                if elapsed > 0:
                    rates.iloc[curr] = change / elapsed
            else:
                gap = curr - prev
                rates.iloc[curr] = change / gap

        valid_rates = rates.dropna()
        if len(valid_rates) < 3:
            return self.copy()

        median_rate = valid_rates.median()
        mad = np.median(np.abs(valid_rates - median_rate))
        if mad < 1e-10:
            return self.copy()

        modified_z = 0.6745 * (rates - median_rate) / mad
        is_outlier = np.abs(modified_z) > threshold

        result = self.copy()
        outlier_idx = np.where(is_outlier & ~original_nan)[0]
        result.iloc[outlier_idx] = np.nan

        result = TimeSeries(result)
        result._processing_history = self._processing_history + [
            f"detect_outliers(threshold={threshold}, removed={len(outlier_idx)})"
        ]
        return result

    # =========================================================================
    # Gap Imputation (SSA)
    # =========================================================================

    def fill_gaps(
        self,
        method: str = "ssa",
        embedding_dim: Optional[int] = None,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.9,
        max_components: Optional[int] = None,
        max_iter: int = 50,
        tol: float = 1e-5,
        smooth_observed: bool = False,
    ) -> "TimeSeries":
        """
        Fill missing values using Singular Spectrum Analysis (SSA).

        Detrends the series, performs iterative SSA with low-rank
        reconstruction, and adds the trend back.

        Parameters
        ----------
        method : str, optional
            Imputation method. Currently only 'ssa'. Default is 'ssa'.
        embedding_dim : int, optional
            Window size (L) for SSA. Auto-tuned if None.
        n_components : int, optional
            Fixed number of SVD components to keep. If None, uses
            variance_threshold.
        variance_threshold : float, optional
            Cumulative variance to retain (0.8 to 0.95). Default is 0.9.
        max_components : int, optional
            Upper limit on components.
        max_iter : int, optional
            Max SSA iterations. Default is 50.
        tol : float, optional
            Convergence tolerance. Default is 1e-5.
        smooth_observed : bool, optional
            If True, smooth ALL values (denoising). If False, only fill
            missing values. Default is False.

        Returns
        -------
        TimeSeries
            Series with gaps filled.

        Examples
        --------
        >>> filled = ts.fill_gaps(variance_threshold=0.9)
        >>> filled_smooth = ts.fill_gaps(smooth_observed=True, n_components=3)

        Notes
        -----
        SSA requires less than 70% missing values. If data has too many
        gaps, consider linear interpolation first.
        """
        if method != "ssa":
            raise ValueError(f"Unknown method '{method}'. Only 'ssa' is supported.")

        series_values = self.values.astype(float).copy()
        n = len(series_values)

        missing_ratio = np.isnan(series_values).mean()
        if missing_ratio > 0.7:
            raise ValueError(
                f"Data has {missing_ratio*100:.1f}% missing values. "
                "SSA requires less than 70% missingness."
            )

        t = np.arange(n)

        # Detrend (linear)
        valid_mask = ~np.isnan(series_values)
        if np.sum(valid_mask) < 2:
            mean_val = np.nanmean(series_values)
            trend_line = np.full(n, mean_val if not np.isnan(mean_val) else 0.0)
        else:
            try:
                coeffs = np.polyfit(t[valid_mask], series_values[valid_mask], 1)
                trend_line = np.polyval(coeffs, t)
            except np.linalg.LinAlgError:
                trend_line = np.full(n, np.nanmean(series_values[valid_mask]))

        detrended = series_values - trend_line

        # Auto-tune embedding dim
        if embedding_dim is None:
            embedding_dim = self._suggest_ssa_parameters(detrended)
        embedding_dim = max(2, min(embedding_dim, n - 1))

        # Run iterative SSA
        imputed_detrended = self._iterative_ssa(
            detrended,
            embedding_dim,
            n_components=n_components,
            variance_threshold=variance_threshold,
            max_components=max_components,
            max_iter=max_iter,
            tol=tol,
            smooth_observed=smooth_observed,
        )

        final = imputed_detrended + trend_line

        result = TimeSeries(final, index=self.index, name=self.name)
        result._processing_history = self._processing_history + [
            f"fill_gaps(method='ssa', embedding_dim={embedding_dim})"
        ]
        return result

    # =========================================================================
    # Sinusoidal Modeling
    # =========================================================================

    def fit_sinusoidal(self, periods: List[float], predict_time: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Fit a sinusoidal model using Linear Regression on Fourier terms.

        This robust method does not require initial guesses for amplitude
        or phase. It linearizes the problem using sin/cos pairs.

        Parameters
        ----------
        periods : list of float
            Periods in time steps (e.g., [365.25, 182.6] for annual +
            semi-annual).
        predict_time : np.ndarray, optional
            Custom time values for prediction. If None, uses the
            numeric index of the series.

        Returns
        -------
        tuple of (np.ndarray, dict)
            (estimation, model_params)
            - estimation: Fitted signal values
            - model_params: dict with 'baseline', 'trend_slope',
              'components' (each with period, amplitude, phase)

        Examples
        --------
        >>> estimation, params = ts.fit_sinusoidal(periods=[365.25])
        >>> print(f"Annual amplitude: {params['components'][0]['amplitude']:.3f}")
        """
        time_values = self.numeric_index().astype(float)
        observed = self.dropna().values

        # Build feature matrix: [Trend, Sin_p1, Cos_p1, Sin_p2, Cos_p2, ...]
        X = time_values.reshape(-1, 1)
        for p in periods:
            omega = 2 * np.pi / p
            X = np.column_stack([X, np.sin(omega * time_values), np.cos(omega * time_values)])

        valid_mask = ~np.isnan(observed)
        if np.sum(valid_mask) < X.shape[1] + 2:
            return None, {}

        reg = LinearRegression()
        reg.fit(X[valid_mask], observed[valid_mask])

        if predict_time is None:
            predict_time = time_values
            X_pred = X
        else:
            X_pred = predict_time.reshape(-1, 1)
            for p in periods:
                omega = 2 * np.pi / p
                X_pred = np.column_stack([
                    X_pred,
                    np.sin(omega * predict_time),
                    np.cos(omega * predict_time),
                ])

        estimation = reg.predict(X_pred)

        # Extract physical parameters
        coeffs = reg.coef_
        params = {
            "baseline": reg.intercept_,
            "trend_slope": coeffs[0],
            "components": [],
        }
        for i, p in enumerate(periods):
            c1 = coeffs[1 + 2 * i]
            c2 = coeffs[1 + 2 * i + 1]
            params["components"].append(
                {
                    "period": p,
                    "amplitude": np.sqrt(c1 ** 2 + c2 ** 2),
                    "phase": np.arctan2(c2, c1),
                }
            )

        return estimation, params

    def prepare_sinusoidal_inputs(self, seasonality_info, select_col=None):
        """
        Prepare input parameters for advanced sinusoidal fitting.

        Parameters
        ----------
        seasonality_info : pd.DataFrame
            DataFrame with columns: Amplitude, Frequency, Phase, Period (days).
            Typically from .find_seasonality().
        select_col : str, optional
            Column name if this is part of a DataFrame operation.

        Returns
        -------
        tuple
            (time_values, observed_values, amplitudes, periods,
             phase_shifts, baseline)

        Examples
        --------
        >>> seasonality = ts.find_seasonality()
        >>> top_3 = seasonality.head(3)
        >>> inputs = ts.prepare_sinusoidal_inputs(top_3)
        """
        required = ["Amplitude", "Frequency", "Phase", "Period (days)"]
        if not all(col in seasonality_info.columns for col in required):
            raise ValueError(f"seasonality_info must have columns: {required}")

        notna = self.notna()
        t = np.arange(len(self))
        t_finite = t[notna]
        y_finite = self.values[notna]

        return (
            t_finite,
            y_finite,
            seasonality_info["Amplitude"].values,
            seasonality_info["Period (days)"].values,
            seasonality_info["Phase"].values,
            np.nanmean(y_finite),
        )

    # =========================================================================
    # Synthetic Data Generation (Class Method)
    # =========================================================================

    @classmethod
    def synthetic(
        cls,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        linear_slope: float = 0.0,
        amplitude_list: Optional[List[float]] = None,
        period_list: Optional[List[float]] = None,
        variance: float = 0.01,
        random_seed: int = 42,
    ) -> "TimeSeries":
        """
        Generate a synthetic time-series with sinusoidal + trend + noise.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format. Default is '2020-01-01'.
        end_date : str
            End date. Default is '2024-12-31'.
        linear_slope : float
            Slope of the linear trend. Default is 0.0.
        amplitude_list : list of float, optional
            Amplitudes of sinusoidal components. Default is [1.0].
        period_list : list of float, optional
            Periods in years. Default is [1.0].
        variance : float
            Noise variance. Default is 0.01.
        random_seed : int
            Random seed. Default is 42.

        Returns
        -------
        TimeSeries
            Synthetic time-series with DatetimeIndex.

        Examples
        --------
        >>> ts = TimeSeries.synthetic(
        ...     amplitude_list=[5.0, 2.0],
        ...     period_list=[1.0, 0.5],
        ...     linear_slope=0.001,
        ...     variance=0.1,
        ... )
        """
        if amplitude_list is None:
            amplitude_list = [1.0]
        if period_list is None:
            period_list = [1.0]

        rng = np.random.default_rng(random_seed)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        days = (dates - dates[0]).days
        pi = np.pi

        seasonal = np.sum(
            [
                amp * np.sin(2 * pi * days / (period * 365.25))
                for amp, period in zip(amplitude_list, period_list)
            ],
            axis=0,
        )
        trend = linear_slope * days
        noise = rng.normal(scale=variance, size=len(dates))

        return cls(seasonal + trend + noise, index=dates, name="synthetic")

    # =========================================================================
    # Private SSA helpers
    # =========================================================================

    @staticmethod
    def _suggest_ssa_parameters(series_values):
        """Auto-tune embedding dimension using FFT peak detection."""
        y = np.array(series_values, dtype=float).copy()
        n = len(y)
        if n < 4:
            return max(2, n // 2)

        missing_ratio = np.isnan(y).mean()
        if missing_ratio > 0.5:
            return max(2, min(n // 4, 20))

        if np.all(np.isnan(y)):
            return max(2, n // 2)

        valid_mask = ~np.isnan(y)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 2:
            return max(2, n // 4)

        y = np.interp(np.arange(n), valid_idx, y[valid_mask])

        # Linear detrend
        x = np.arange(n)
        try:
            coeffs = np.polyfit(x, y, 1)
            y_detrended = y - np.polyval(coeffs, x)
        except np.linalg.LinAlgError:
            y_detrended = y - np.mean(y)

        yf = rfft(y_detrended)
        xf = rfftfreq(n, 1.0)
        if len(xf) <= 1:
            return max(2, n // 4)

        magnitudes = np.abs(yf[1:])
        if magnitudes.size == 0:
            return max(2, n // 4)

        peak_idx = np.argsort(magnitudes)[-3:]
        slowest_freq = np.min(xf[1:][peak_idx])

        nyquist = 0.5
        if slowest_freq <= nyquist / n:
            dominant_period = n / 4.0
        else:
            dominant_period = 1.0 / slowest_freq

        suggested = int(max(dominant_period * 1.5, n / 4.0))
        suggested = min(suggested, int(n / 3.0))
        suggested = min(suggested, 100)
        suggested = max(2, min(suggested, n - 1))
        return suggested

    @staticmethod
    def _iterative_ssa(
        series, embedding_dim, n_components=None,
        variance_threshold=0.9, max_components=None,
        max_iter=30, tol=1e-5, smooth_observed=False
    ):
        """Iterative SSA solver with variance-based component selection."""
        current = np.array(series, dtype=float)
        missing = np.isnan(current)

        if np.all(missing):
            return np.zeros_like(current)

        # Normalize
        valid = ~missing
        mean = np.mean(current[valid])
        std = np.std(current[valid])
        if std < 1e-10:
            std = 1.0
        current = (current - mean) / std
        current[missing] = 0.0

        prev = current.copy()
        n = len(current)

        for iteration in range(max_iter):
            # Embedding (Hankel matrix)
            num_vec = n - embedding_dim + 1
            if num_vec <= 0:
                break
            X = np.full((num_vec, embedding_dim), np.nan)
            for i in range(embedding_dim):
                X[:, i] = current[i : i + num_vec]

            X_mean = np.nanmean(X, axis=0)
            X_centered = X - X_mean

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                if iteration == 0:
                    raise RuntimeError("SVD failed. Data may be ill-conditioned.")
                return prev * std + mean

            # Select components
            if n_components is None:
                variance = S ** 2
                cum_var = np.cumsum(variance / variance.sum())
                exceeded = cum_var >= variance_threshold
                r = (np.argmax(exceeded) + 1) if np.any(exceeded) else len(S)
                if max_components is not None:
                    r = min(r, max_components)
                r = max(1, min(r, len(S) // 2 if len(S) > 2 else len(S)))
            else:
                r = min(int(n_components), len(S))
                r = min(r, embedding_dim - 1 if embedding_dim > 1 else 1)

            # Low-rank reconstruction
            X_recon = (U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]) + X_mean

            # Diagonal averaging (vectorized)
            rows, cols = np.indices(X_recon.shape)
            positions = rows + cols
            recon = np.zeros(n)
            counts = np.zeros(n)
            np.add.at(recon, positions.ravel(), X_recon.ravel())
            np.add.at(counts, positions.ravel(), 1.0)
            valid = counts > 0
            recon[valid] /= counts[valid]
            recon[~valid] = np.nan

            # Update
            if smooth_observed:
                diff = np.linalg.norm(recon - prev)
                current[:] = recon
            else:
                diff = np.linalg.norm(recon[missing] - prev[missing])
                current[missing] = recon[missing]

            if diff < tol and iteration > 0:
                break
            prev = current.copy()

        return current * std + mean
