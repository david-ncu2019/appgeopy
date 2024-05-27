import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor

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
