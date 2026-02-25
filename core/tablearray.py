"""
TableArray - A pandas DataFrame subclass for tabular data operations.

Extends pd.DataFrame with convenience methods for Excel/JSON/CSV I/O,
datetime alignment, and time-series extraction.

Examples
--------
>>> from appgeopy.core import TableArray
>>> ta = TableArray.from_csv('data.csv')
>>> ta = TableArray.from_excel('data.xlsx', sheet_name='Sheet1')
>>> help(TableArray)
"""

from __future__ import annotations

import json
import os
import shutil
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .timeseries import TimeSeries


class TableArray(pd.DataFrame):
    """
    A DataFrame subclass with integrated I/O and data extraction methods.

    Inherits all pandas.DataFrame functionality. Adds convenience methods
    for file I/O (Excel, CSV, JSON), datetime alignment, and extracting
    single-column TimeSeries objects.

    Examples
    --------
    >>> ta = TableArray({'date': dates, 'value': values})
    >>> ta = TableArray.from_excel('stations.xlsx', sheet_name='GPS')
    >>> ts = ta.extract_timeseries('displacement', index_col='date')
    """

    _metadata = ["_source_file"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_source_file"):
            self._source_file = None

    @property
    def _constructor(self):
        return TableArray

    @property
    def _constructor_sliced(self):
        return TimeSeries

    # =========================================================================
    # File I/O: Loading
    # =========================================================================

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        """
        Load a TableArray from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        **kwargs
            Additional arguments passed to pd.read_csv().

        Returns
        -------
        TableArray

        Examples
        --------
        >>> ta = TableArray.from_csv('data.csv', parse_dates=['date'])
        >>> ta = TableArray.from_csv('data.csv', index_col=0)
        """
        df = pd.read_csv(filepath, **kwargs)
        result = cls(df)
        result._source_file = filepath
        return result

    @classmethod
    def from_excel(cls, filepath, sheet_name=0, **kwargs):
        """
        Load a TableArray from an Excel file.

        Parameters
        ----------
        filepath : str
            Path to the Excel file (.xlsx).
        sheet_name : str or int, optional
            Sheet name or index. Default is 0 (first sheet).
        **kwargs
            Additional arguments passed to pd.read_excel().

        Returns
        -------
        TableArray

        Examples
        --------
        >>> ta = TableArray.from_excel('stations.xlsx', sheet_name='data')
        """
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        result = cls(df)
        result._source_file = filepath
        return result

    @classmethod
    def from_json(cls, filepath):
        """
        Load a TableArray from a JSON file.

        The JSON file should contain a dictionary that can be converted
        to a DataFrame. For nested dictionaries, use read_json_to_dict()
        instead.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        TableArray

        Examples
        --------
        >>> ta = TableArray.from_json('data.json')
        """
        df = pd.read_json(filepath)
        result = cls(df)
        result._source_file = filepath
        return result

    # =========================================================================
    # File I/O: Saving
    # =========================================================================

    def to_excel_sheet(
        self, filepath, sheet_name, mode="a", if_sheet_exists="replace",
        index=False, verbose=True
    ):
        """
        Save to a specific sheet in an Excel file.

        Handles creating new files, appending sheets, and replacing
        existing sheets.

        Parameters
        ----------
        filepath : str
            Excel file path.
        sheet_name : str
            Target sheet name.
        mode : str, optional
            'a' to append, 'w' to overwrite the file. Default is 'a'.
        if_sheet_exists : str, optional
            'replace', 'new', or 'skip'. Default is 'replace'.
        index : bool, optional
            Include DataFrame index. Default is False.
        verbose : bool, optional
            Print status messages. Default is True.

        Examples
        --------
        >>> ta.to_excel_sheet('output.xlsx', 'results')
        >>> ta.to_excel_sheet('output.xlsx', 'raw', mode='w')
        """
        try:
            file_exists = os.path.isfile(filepath)
            if not file_exists or mode == "w":
                with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
                    self.to_excel(writer, sheet_name=sheet_name, index=index)
                if verbose:
                    print(f"Created '{filepath}', sheet '{sheet_name}'.")
                return

            with pd.ExcelWriter(
                filepath, engine="openpyxl", mode="a",
                if_sheet_exists=if_sheet_exists
            ) as writer:
                self.to_excel(writer, sheet_name=sheet_name, index=index)
            if verbose:
                print(f"Written to sheet '{sheet_name}' in '{filepath}'.")
        except Exception as e:
            print(f"Error writing Excel: {e}")

    def to_json_file(self, folder_path, file_name, indent=4, sort_keys=True,
                     overwrite=True):
        """
        Save the DataFrame as a JSON file.

        Parameters
        ----------
        folder_path : str
            Directory to save the file.
        file_name : str
            File name (without extension).
        indent : int, optional
            JSON indentation level. Default is 4.
        sort_keys : bool, optional
            Sort dictionary keys. Default is True.
        overwrite : bool, optional
            Overwrite existing file. Default is True.

        Returns
        -------
        str
            Path to the saved JSON file.

        Examples
        --------
        >>> path = ta.to_json_file('./output', 'results')
        """
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{file_name}.json")

        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(f"File '{file_path}' already exists.")

        data_dict = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(data_dict, f, indent=indent, sort_keys=sort_keys, default=str)
        return file_path

    # =========================================================================
    # Static I/O helpers
    # =========================================================================

    @staticmethod
    def get_sheet_names(filepath):
        """
        List all sheet names of an Excel file.

        Parameters
        ----------
        filepath : str
            Path to the Excel file (.xlsx).

        Returns
        -------
        list of str or None
            List of sheet names, or None if file is invalid.

        Examples
        --------
        >>> sheets = TableArray.get_sheet_names('data.xlsx')
        >>> print(sheets)
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        if not filepath.lower().endswith(".xlsx"):
            print(f"Not an Excel file: {filepath}")
            return None
        try:
            return pd.ExcelFile(filepath).sheet_names
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return None

    @staticmethod
    def read_json_to_dict(filepath):
        """
        Read a JSON file and return as a Python dictionary.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        dict

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not valid JSON.

        Examples
        --------
        >>> config = TableArray.read_json_to_dict('config.json')
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: '{filepath}'")
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in '{filepath}': {e}")
        return data

    @staticmethod
    def save_dict_to_json(dictionary, folder_path, file_name, indent=4,
                          sort_keys=True, overwrite=True):
        """
        Save a Python dictionary to a JSON file.

        Parameters
        ----------
        dictionary : dict
            Dictionary to save.
        folder_path : str
            Directory to save the file.
        file_name : str
            File name (without extension).
        indent : int, optional
            Indentation level. Default is 4.
        sort_keys : bool, optional
            Sort keys alphabetically. Default is True.
        overwrite : bool, optional
            Overwrite existing file. Default is True.

        Returns
        -------
        str
            Path to the saved file.

        Examples
        --------
        >>> path = TableArray.save_dict_to_json({'a': 1}, './out', 'config')
        """
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary.")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{file_name}.json")
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(f"File '{file_path}' already exists.")
        with open(file_path, "w") as f:
            json.dump(dictionary, f, indent=indent, sort_keys=sort_keys)
        return file_path

    # =========================================================================
    # Time-Series Extraction
    # =========================================================================

    def extract_timeseries(self, column, index_col=None):
        """
        Extract a single column as a TimeSeries object.

        Parameters
        ----------
        column : str
            Column name to extract.
        index_col : str, optional
            Column to use as the datetime index. If None, uses the
            existing DataFrame index.

        Returns
        -------
        TimeSeries

        Examples
        --------
        >>> ts = ta.extract_timeseries('displacement', index_col='date')
        >>> trend, slope = ts.get_trend()
        """
        if index_col is not None:
            idx = pd.to_datetime(self[index_col])
            return TimeSeries(self[column].values, index=idx, name=column)
        return TimeSeries(self[column])

    # =========================================================================
    # DateTime Alignment
    # =========================================================================

    def align_to_fulltime(self, freq="D"):
        """
        Extend the DataFrame to cover the full date range in the index.

        Fills missing dates with NaN rows.

        Parameters
        ----------
        freq : str, optional
            Frequency string. Default is 'D' (daily).

        Returns
        -------
        TableArray
            DataFrame with complete date range.

        Raises
        ------
        ValueError
            If the index is not a DatetimeIndex.

        Examples
        --------
        >>> ta_full = ta.align_to_fulltime(freq='D')
        """
        if not isinstance(self.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        full_range = pd.date_range(self.index[0], self.index[-1], freq=freq)
        missing = [d for d in full_range if d not in self.index]

        if not missing:
            return self.copy()

        missing_df = pd.DataFrame(
            np.nan, index=missing, columns=self.columns
        )
        combined = pd.concat([self, missing_df]).sort_index()
        return TableArray(combined)

    def cumulative_to_incremental(self, datetime_columns):
        """
        Convert cumulative values to incremental (displacement) values.

        Computes the difference between consecutive datetime columns.
        The first column retains its original value.

        Parameters
        ----------
        datetime_columns : list of str
            Column names representing consecutive time steps.

        Returns
        -------
        TableArray
            DataFrame with incremental values.

        Examples
        --------
        >>> incremental = ta.cumulative_to_incremental(date_cols)
        """
        df_dates = self.loc[:, datetime_columns]
        displacement = df_dates - df_dates.shift(1, axis=1)
        displacement.iloc[:, 0] = df_dates.iloc[:, 0]

        idx_first = list(self.columns).index(datetime_columns[0])
        result = pd.concat([self.iloc[:, :idx_first], displacement], axis=1)
        return TableArray(result)

    # =========================================================================
    # File Backup
    # =========================================================================

    @staticmethod
    def save_config_file(src_file, dest_folder, step=None, verbose=True):
        """
        Copy a file to a destination folder (overwrites only if newer).

        Parameters
        ----------
        src_file : str
            Path to the source file.
        dest_folder : str
            Destination directory.
        step : str, optional
            If 'evaluation', 'calibration', or 'validation', renames
            the file accordingly.
        verbose : bool, optional
            Print status messages. Default is True.

        Returns
        -------
        str
            Message describing the action taken.

        Examples
        --------
        >>> TableArray.save_config_file('config.ini', './backup/')
        """
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Source file not found: '{src_file}'")

        os.makedirs(dest_folder, exist_ok=True)
        dest_file = os.path.join(dest_folder, os.path.basename(src_file))

        step_names = {
            "evaluation": "evaluation_config.ini",
            "calibration": "calibration_config.ini",
            "validation": "validation_config.ini",
        }
        if step in step_names:
            dest_file = os.path.join(dest_folder, step_names[step])

        if os.path.exists(dest_file):
            if os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                shutil.copy2(src_file, dest_file)
                msg = f"Overwritten: '{dest_file}' with newer '{src_file}'."
            else:
                msg = f"Up-to-date: '{dest_file}'. No action taken."
        else:
            shutil.copy2(src_file, dest_file)
            msg = f"Saved: '{src_file}' to '{dest_folder}'."

        if verbose:
            print(msg)
        return msg
