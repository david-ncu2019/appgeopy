import numpy as np
import pandas as pd


class TimeSeries:
    """Base timeseries data structure and operations"""

    def __init__(self, data, index=None):
        if isinstance(data, pd.Series):
            self.data = data
        else:
            self.data = pd.Series(data, index=index)

    # Move from datetime_handle.py:
    def get_fulltime(self, freq="D"):
        try:
            start_time = series[0]
            end_time = series[-1]
            fulltime = pd.date_range(start_time, end_time, freq=freq)
            return fulltime
        except Exception as e:
            raise ValueError(f"{e}")

    def fulltime_table(self, fulltime_series):
        """
        Align a pandas Series/DataFrame to a full timeline by inserting NaNs for missing dates.

        Parameters:
            df (pd.Series or pd.DataFrame): Input data with datetime index.
            fulltime_series (pd.DatetimeIndex): Complete reference time index.

        Returns:
            pd.Series or pd.DataFrame: Data aligned to full timeline, preserving input type.
        """
        # Check if input is a Series (store metadata to restore later)
        is_series = isinstance(df, pd.Series)
        original_name = df.name if is_series else None

        # Convert Series → DataFrame to unify processing
        if is_series:
            df = df.to_frame(name=original_name if original_name else "value")

        # Validate index types match
        if not isinstance(df.index, type(fulltime_series)):
            raise ValueError(
                "Index type of `df` must match `fulltime_series` (e.g., DatetimeIndex)"
            )

        # Get missing dates (using Pandas index operations)
        missing_dates = fulltime_series.difference(df.index)

        # Create NaN entries for missing dates
        null_table = pd.DataFrame(
            columns=df.columns,
            index=missing_dates,
            dtype=df.dtypes.iloc[0] if not df.empty else None,
        )

        # Concatenate and sort
        combined = pd.concat([df, null_table]).sort_index()

        # Convert back to Series if input was a Series
        if is_series:
            combined = combined.squeeze().rename(original_name)

        return combined

    def intersect_time_index(self, other_index):
        """
        Finds the intersection of two time indices.

        Args:
            df1_index (iterable): An iterable of time indices (e.g., list, set, pandas Index) for the first dataset.
            df2_index (iterable): An iterable of time indices (e.g., list, set, pandas Index) for the second dataset.

        Returns:
            list: A sorted list of the common elements in both time indices.

        Raises:
            TypeError: If either input is not an iterable.
            ValueError: If either input is empty.
        """
        try:
            # Ensure inputs are iterables that can be converted to sets
            _a = set(df1_index)
            _b = set(df2_index)
        except TypeError as e:
            raise TypeError(
                "Both inputs must be iterables (e.g., list, set, pandas Index)."
            ) from e

        if not _a:
            raise ValueError("The first input time index is empty.")
        if not _b:
            raise ValueError("The second input time index is empty.")

        # Find intersection and sort the result
        intersection = sorted(list(_a.intersection(_b)))

        return intersection

    def numeric_time_index(self):
        """
        Generate a numeric time index for a given time series, excluding null values.

        Parameters:
            time_series (pandas.Series): A pandas Series with a DatetimeIndex, which may contain null values.

        Returns:
            numpy.ndarray: An array of numeric indices corresponding to the non-null values in the input time series.
        """
        # Create a boolean filter for non-null values in the time series
        non_null_filter = time_series.notna()

        # Generate a numeric array representing the time indices
        numeric_time_array = np.arange(len(time_series))

        # Apply the non-null filter to the numeric time array
        numeric_time_array_finite = numeric_time_array[non_null_filter]

        return numeric_time_array_finite

    # Move from smoothing.py:
    def simple_moving_average(self, window_size=7):
        """
        Calculate the simple moving average of an input array with a given window size.

        Parameters:
            num_arr (array-like): Input array or list of numerical values.
            window_size (int, optional): The size of the moving window. Default is 7.

        Returns:
            list: A list containing the moving averages. If a window contains NaN values, the average will ignore them.

        Example:
            >>> simple_moving_average([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, np.nan], window_size=3)
            [1.3333333333333333, 1.5, 3.0, 4.5, 4.5, 6.0, 7.5, 8.0, 8.5, 9.0]
        """
        half_window = window_size // 2

        # Handle different input types: list, NumPy array, or Pandas Series
        first_element = (
            num_arr[0]
            if isinstance(num_arr, (list, np.ndarray))
            else num_arr.iloc[0]
        )
        last_element = (
            num_arr[-1]
            if isinstance(num_arr, (list, np.ndarray))
            else num_arr.iloc[-1]
        )

        padded_arr = (
            [first_element] * half_window
            + list(num_arr)
            + [last_element] * half_window
        )

        # Sliding window generator
        def sliding_window(iterable, size):
            it = iter(iterable)
            result = tuple(islice(it, size))
            if len(result) == size:
                yield result
            for elem in it:
                result = result[1:] + (elem,)
                yield result

    # Move from interpolate.py:
    def spline_interp(self, factor=10, k=1):
        """
        Smoothly interpolate data using a B-spline.

        Parameters:
        -----------
        - x (np.ndarray): 1D array of sorted numeric values (e.g., depth or time).
        - y (np.ndarray): 1D array of numeric values corresponding to `x`
        - factor (int): The number of points to generate between min and max x.
        - k (int) : optional (default=2)
            Degree of the spline. Must be 1 (linear), 2 (quadratic), 3 (cubic), etc.
            Higher values create smoother curves but may overfit with limited data points.

        Returns:
        -------
        - np.ndarray: Interpolated x-values with `len(x) * factor` points.
        - np.ndarray: Interpolated y-values corresponding to `x_new`.
        """
        # Validate that k is within an acceptable range (1 to 5)
        if not (1 <= k <= 5):
            raise ValueError(
                "Parameter 'k' must be an integer between 1 and 5 for smooth interpolation."
            )

        # Perform interpolation using a B-spline of degree k
        spline = make_interp_spline(x, y, k=k)
        x_new = np.linspace(x.min(), x.max(), len(x) * factor)
        y_new = spline(x_new)
        return x_new, y_new