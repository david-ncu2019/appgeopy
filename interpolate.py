from scipy.interpolate import make_interp_spline
import numpy as np

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Optimized quadratic interpolation function using spline
def spline_interp(x, y, factor=10, k=1):
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
        raise ValueError("Parameter 'k' must be an integer between 1 and 5 for smooth interpolation.")

    # Perform interpolation using a B-spline of degree k
    spline = make_interp_spline(x, y, k=k)
    x_new = np.linspace(x.min(), x.max(), len(x) * factor)
    y_new = spline(x_new)
    return x_new, y_new

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --