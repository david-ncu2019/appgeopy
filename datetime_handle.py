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
