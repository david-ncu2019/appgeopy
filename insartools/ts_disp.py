from numpy import pi, sin, cos

def degree_to_radian(degree):
    rad = degree * pi / 180
    return rad

# ------------------------------------------------------------------------------

def get_LOS_disp(dN, dE, dU, incidence_angle=37, heading_angle=347.6):
    """
    Calculates the line-of-sight (LOS) displacement for a given set of north, east, and up displacement components, as well as an incidence angle and heading angle.

    Args:
    dN (float): North displacement component in meters
    dE (float): East displacement component in meters
    dU (float): Up displacement component in meters
    incidence_angle (float): Incidence angle in degrees (default=37)
    heading_angle (float): Heading angle in degrees (default=347.6)

    Returns:
    float: The LOS displacement in meters
    """
    from numpy import cos, pi, sin

    incidence_rad = degree_to_radian(incidence_angle)
    azi_rad = degree_to_radian(heading_angle)

    LOS_disp = (
        dU * cos(incidence_rad)
        + dN * sin(incidence_rad) * sin(azi_rad)
        - dE * sin(incidence_rad) * cos(azi_rad)
    )

    return LOS_disp

# ------------------------------------------------------------------------------

def compare_LOS_disp(psc_df, gps_df, mutual_index):
    """
    Compare Line-of-Sight (LOS) displacements between Persistent Scatterers (PSInSAR) and GPS measurements.

    Args:
        psc_df (pd.DataFrame): DataFrame containing displacements of persistent scatterers (PSInSAR).
        gps_df (pd.DataFrame): DataFrame containing GPS station measurements.
        mutual_index (pd.Index or list): Index or list of indices that both `psc_df` and `gps_df` have in common.

    Returns:
        pd.DataFrame: A DataFrame with the difference in LOS displacements between GPS and PSInSAR measurements for the mutual index.

    Raises:
        ValueError: If `mutual_index` is not in both DataFrames.
        TypeError: If inputs are not valid pandas DataFrames or if `mutual_index` is not a valid index type.
    """
    import pandas as pd

    # Type checking
    if not isinstance(psc_df, pd.DataFrame) or not isinstance(gps_df, pd.DataFrame):
        raise TypeError("Both `psc_df` and `gps_df` must be pandas DataFrames.")
    
    if not isinstance(mutual_index, (pd.Index, list)):
        raise TypeError("`mutual_index` must be a pandas Index or a list of indices.")

    # Check if the mutual index exists in both DataFrames
    if not set(mutual_index).issubset(psc_df.index) or not set(mutual_index).issubset(gps_df.index):
        raise ValueError("The `mutual_index` must be present in both `psc_df` and `gps_df`.")

    # Ensure no missing values in the mutual index for both DataFrames
    mutual_index = pd.Index(mutual_index)
    psc_by_idx = psc_df.loc[mutual_index]
    gps_by_idx = gps_df.loc[mutual_index]

    if psc_by_idx.empty or gps_by_idx.empty:
        raise ValueError("No common index found after dropping missing values.")

    # Calculate the difference in LOS displacements
    diff = gps_by_idx - psc_by_idx

    return diff

# ------------------------------------------------------------------------------

def convert_cumdisp_to_disp(cumulative_series):
    """
    Convert a series of cumulative displacement values into individual displacement values.
    
    Each element in the returned series represents the displacement from the previous element.
    The first value remains unchanged, as there is no preceding element to calculate the difference.

    Args:
    - cumulative_series (pd.Series): A pandas Series representing cumulative displacement values.

    Returns:
    - pd.Series: A new Series where the first value is the same as in the input series,
                 and each subsequent value is the difference between the current and the previous value.
    """
    # Create a copy of the input series to avoid modifying the original data.
    displacement_series = cumulative_series.copy()

    # Shift the series up by one index to align current and previous values.
    shifted_series = cumulative_series.shift(-1)

    # Calculate the difference between the shifted series and the original series.
    differences = shifted_series - cumulative_series

    # Set the first value of the displacement series as the first value of the cumulative series.
    # Assign the calculated differences to the displacement series, skipping the first element.
    displacement_series[1:] = differences[:-1]

    return displacement_series

