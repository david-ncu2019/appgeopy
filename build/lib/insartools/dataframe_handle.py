import shapefile as sf
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------

def pointkey_from_coordinates(xcoord_arr, ycoord_arr):
    """
    Generate a list of point keys from arrays of x and y coordinates.

    Args:
    - xcoord_arr (array-like): An array of x coordinates.
    - ycoord_arr (array-like): An array of y coordinates.

    Returns:
    - list: A list of strings where each string is a combination of the x and y coordinates in the format "X<value>Y<value>".
    """
    # Zip the x and y coordinates and format them into strings "X<value>Y<value>"
    pointkey_list = ["X{}Y{}".format(int(x), int(y)) for x, y in zip(xcoord_arr, ycoord_arr)]
    return pointkey_list

# ------------------------------------------------------------------------------

def modify_ras2ptatt_shp(shp_fpath, datetime_list, xcoord_colname, ycoord_colname):
    """
    Modify the RAS2PTATT output shapefile to add new fields and compute PointKeys.

    Args:
    - shp_fpath (str): Path to the RAS2PTATT output shapefile.
    - datetime_list (list): A list of datetime strings representing the measurement dates.
    - xcoord_colname (str): The column name for the x coordinates.
    - ycoord_colname (str): The column name for the y coordinates.

    Returns:
    - pd.DataFrame: A DataFrame with updated field names and a new 'PointKey' column.
    """
    # ------------------------------------------------------------------------------
    # Read the shapefile and process its records
    # ------------------------------------------------------------------------------
    # Open and read the shapefile using the shapefile library
    with sf.Reader(shp_fpath) as shp_file:
        # Extract the records from the shapefile
        shapefile_records = shp_file.records()

    # Convert the records to a NumPy array for easy manipulation
    records_array = np.array(shapefile_records, dtype=np.float64)

    # ------------------------------------------------------------------------------
    # Create new field names for the modified DataFrame
    # ------------------------------------------------------------------------------
    # Combine the x and y column names with the datetime strings, excluding the initial date
    fieldnames = [xcoord_colname, ycoord_colname] + datetime_list

    # ------------------------------------------------------------------------------
    # Create a dictionary to store the records with new field names
    # ------------------------------------------------------------------------------
    # Construct a dictionary where keys are new field names and values are corresponding columns from the records
    record_dict = {
        fieldname: records_array[:, i]
        for i, fieldname in enumerate(fieldnames)
        if i < records_array.shape[1]
    }

    # ------------------------------------------------------------------------------
    # Convert the dictionary to a DataFrame
    # ------------------------------------------------------------------------------
    # Create a pandas DataFrame from the dictionary
    modified_cumdisp = pd.DataFrame(record_dict)

    # Generate PointKeys from x and y coordinates
    pointkey_col = pointkey_from_coordinates(modified_cumdisp[xcoord_colname], modified_cumdisp[ycoord_colname])

    # Insert the 'PointKey' column at the beginning of the DataFrame
    modified_cumdisp.insert(loc=0, column="PointKey", value=pointkey_col)

    return modified_cumdisp

# ------------------------------------------------------------------------------

def cumdisp_to_disp_dataframe(cumdisp_dataframe, datetime_list):
    """
    Convert a cumulative displacement DataFrame into a displacement DataFrame.

    Args:
    - cumdisp_dataframe (pd.DataFrame): DataFrame containing cumulative displacement values.
    - datetime_list (list): A list of datetime strings representing the measurement dates.

    Returns:
    - pd.DataFrame: A DataFrame where each entry represents the displacement between consecutive measurements.
    """
    # Select columns corresponding to the provided datetime list
    df_by_datetime = cumdisp_dataframe.loc[:, datetime_list]

    displacement_df = df_by_datetime - df_by_datetime.shift(1, axis=1)
    displacement_df.iloc[:, 0] = df_by_datetime.iloc[:, 0]

    # Find the index of the first date column in the DataFrame
    idx_first_date = list(cumdisp_dataframe.columns).index(datetime_list[0])

    # Concatenate the displacement DataFrame with the initial part of the cumulative displacement DataFrame
    return pd.concat([cumdisp_dataframe.iloc[:, :idx_first_date], displacement_df], axis=1)

# ------------------------------------------------------------------------------

def convert_to_fulltime(dataframe):
    """
    Extend a DataFrame to include all dates within the range from the first to the last date of the index,
    filling any missing dates with NaN values.

    Parameters:
    - dataframe (pd.DataFrame): The original DataFrame with a DateTime index.

    Returns:
    - pd.DataFrame: A new DataFrame with a complete range of dates from start to end, including the original data 
      and filling missing dates with NaN.
    
    Raises:
    - ValueError: If the DataFrame does not have a DateTime index.
    - Exception: For other general exceptions.
    """
    try:
        # Ensure the DataFrame has a DateTime index
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame must have a DateTime index.")

        # Get the existing date range from the DataFrame
        existing_datetime_index = dataframe.index
        start_date = existing_datetime_index[0]
        end_date = existing_datetime_index[-1]

        # Generate the full range of dates from the start to end
        full_datetime_index = pd.date_range(start=start_date, end=end_date)

        # Identify missing dates
        missing_dates = [date for date in full_datetime_index if date not in existing_datetime_index]

        # Create a DataFrame for the missing dates, filled with NaN
        missing_dates_df = pd.DataFrame(data=np.nan, index=missing_dates, columns=dataframe.columns)

        # Concatenate the original DataFrame with the missing dates DataFrame
        extended_dataframe = pd.concat([dataframe, missing_dates_df])

        # Sort the DataFrame by the index to maintain chronological order
        extended_dataframe.sort_index(inplace=True)

        return extended_dataframe

    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# ------------------------------------------------------------------------------