import pandas as pd
import numpy as np
import sys

def get_fulltime(series, freq='D'):
	try:
		start_time = series[0]
		end_time = series[-1]
		fulltime = pd.date_range(start_time, end_time, freq=freq)
		return fulltime
	except Exception as e:
		raise ValueError(f"{e}")


def fulltime_table(df, fulltime_series):
	if type(df.index[0])==type(fulltime_series[0]):
		null_table = pd.DataFrame(
			data=None,
			columns=df.columns,
			index=fulltime_series
			)
		_merge = pd.concat([df, null_table])
		_merge = _merge.sort_index()
		return _merge
	else:
		sys.exit("Data types of DataFrame indexes and input series do not match")

def convert_to_datetime(colname):
	if "N" in colname:
		colname = colname[1:]

	return pd.to_datetime(colname)


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

