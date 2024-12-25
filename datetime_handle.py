import pandas as pd
import numpy as np
import sys
import os
import re
from datetime import datetime

def get_fulltime(series, freq='D'):
	try:
		start_time = series[0]
		end_time = series[-1]
		fulltime = pd.date_range(start_time, end_time, freq=freq)
		return fulltime
	except Exception as e:
		raise ValueError(f"{e}")

# ------------------------------------------------------------------------------

def fulltime_table(df, fulltime_series):
    """
    Create a DataFrame that combines the input DataFrame with the missing dates from a full time series.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with a datetime index.
    - fulltime_series (pd.DatetimeIndex): A series of dates representing the complete time range.

    Returns:
    - pd.DataFrame: A DataFrame with the full time series, including null rows for missing dates.
    
    Raises:
    - ValueError: If the data types of the DataFrame index and full time series do not match.
    """
    if isinstance(df.index[0], type(fulltime_series[0])):
        # Calculate the remaining dates after removing those already in the df index.
        df_indexes = set(df.index)
        fulltime_series = set(fulltime_series)
        remaining_dates = sorted(fulltime_series.difference(df_indexes))
        
        # Create a null DataFrame with the remaining dates as its index.
        null_table = pd.DataFrame(
            data=None,
            columns=df.columns,
            index=remaining_dates
        )
        
        # Concatenate the original DataFrame with the null table and sort by index.
        combined_df = pd.concat([df, null_table]).sort_index()
        return combined_df
    else:
        raise ValueError("Data types of DataFrame index and input series do not match")

# ------------------------------------------------------------------------------

def convert_to_datetime(colname):
    """
    Convert a string to a datetime object based on specific validation rules.

    Parameters:
    - colname (str): The input string to be converted.

    Returns:
    - datetime: A datetime object representing the date.

    Raises:
    - ValueError: If the input string does not meet the required format.
    """
    # Check if the input contains alphabetical characters
    match = re.search(r'[A-Za-z]', colname)
    
    if match:
        # Get the last alphabetical character's index and check the remaining string
        last_alpha_idx = match.end() - 1
        
        # Extract the part after the last alphabetical character
        numeric_part = colname[last_alpha_idx + 1:]
        
        # Check if the remaining part has exactly 8 digits
        if len(numeric_part) == 8 and numeric_part.isdigit():
            return pd.to_datetime(numeric_part, format='%Y%m%d')
        else:
            raise ValueError("Input must end with 'YYYYMMDD' after letters.")
    
    # If no alphabetical characters, check if the string is 8 digits
    elif colname.isdigit() and len(colname) == 8:
        return pd.to_datetime(colname, format='%Y%m%d')
    
    else:
        raise ValueError("Input must be 'YYYYMMDD' or contain letters followed by 'YYYYMMDD'.")

# ------------------------------------------------------------------------------

def datetime_to_string(date, initial_char='N'):
    """
    Convert a datetime object to a formatted string with an optional prefix.

    Parameters:
    - date (datetime): The datetime object to convert.
    - initial_char (str): The characters to prefix the date string. Default is 'N'.

    Returns:
    - str: The formatted date string in the form of initial_char + 'YYYYMMDD'.

    Raises:
    - ValueError: If the input initial_char contains non-alphabetical characters.
    - TypeError: If the date is not a datetime object.
    """
    # Validate input
    if not isinstance(initial_char, str) or not initial_char.isalpha():
        raise ValueError("Initial character(s) must only contain alphabetical characters.")
    if not isinstance(date, (pd.Timestamp, pd.DatetimeIndex, pd.Timestamp)):
        raise TypeError("The date must be a datetime object.")
    
    # Convert datetime to string in 'YYYYMMDD' format
    date_str = date.strftime('%Y%m%d')
    
    # Combine initial character with the formatted date
    return f"{initial_char}{date_str}"

# ------------------------------------------------------------------------------

def intersect_time_index(df1_index, df2_index):
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
        raise TypeError("Both inputs must be iterables (e.g., list, set, pandas Index).") from e

    if not _a:
        raise ValueError("The first input time index is empty.")
    if not _b:
        raise ValueError("The second input time index is empty.")
    
    # Find intersection and sort the result
    intersection = sorted(list(_a.intersection(_b)))

    return intersection

# ------------------------------------------------------------------------------

def extract_datetime_from_mfile(mfile):
    """
    Extract unique datetime components from filenames in a specified mfile.

    Parameters:
    - mfile (str): Path to the mfile containing list of filenames.

    Returns:
    - list: Sorted list of unique datetime components extracted from filenames.
    """
    
    # Read lines from the mfile and strip whitespace/newline characters
    with open(mfile, "r") as input_file:
        lines = [line.strip() for line in input_file]
    
    # Extract basenames (without extension) from the lines
    basenames = [os.path.basename(os.path.splitext(line)[0]) for line in lines]
    
    # Use a set comprehension to collect unique datetime components
    datetimes = {"N"+name.split("_")[-2][3:] for name in basenames}.union({"N"+name.split("_")[-1][3:] for name in basenames})
    
    # Return a sorted list of unique datetime components
    return sorted(datetimes)

# ------------------------------------------------------------------------------

def numeric_time_index(time_series):
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

# ------------------------------------------------------------------------------

def bytes_to_datetime(byte_string, date_format='%Y%m%d'):
    """
    Convert a byte string representing a date into a datetime object.

    Parameters:
    - byte_string (bytes): A byte string that contains a date in a specific format.
                           For example, b'20041123' for the date '2004-11-23'.
    - date_format (str): The format of the date in the byte string. Default is '%Y%m%d',
                         which corresponds to 'YYYYMMDD'.

    Returns:
    - datetime: A datetime object representing the date encoded in the byte string.

    Raises:
    - TypeError: If the input is not a byte string.
    - ValueError: If the byte string cannot be decoded into a string, or if the date string
                  does not match the expected format.
    
    Example:
    >>> bytes_to_datetime(b'20041123')
    datetime.datetime(2004, 11, 23, 0, 0)

    >>> bytes_to_datetime(b'23-11-2004', '%d-%m-%Y')
    datetime.datetime(2004, 11, 23, 0, 0)
    """
    # Check if the input is a byte string
    if not isinstance(byte_string, bytes):
        raise TypeError("Input must be of type 'bytes', but got type '{}'.".format(type(byte_string).__name__))

    try:
        # Decode the byte string to a regular string using UTF-8 encoding
        date_string = byte_string.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError("Failed to decode byte string. Ensure the byte string is properly encoded in UTF-8.") from e

    try:
        # Convert the string to a datetime object using the specified format
        date_obj = datetime.strptime(date_string, date_format)
    except ValueError as e:
        raise ValueError("The date string '{}' does not match the expected format '{}'."
                         " Please ensure the format corresponds to the date structure.".format(date_string, date_format)) from e

    return date_obj

# ------------------------------------------------------------------------------