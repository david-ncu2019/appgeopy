import pandas as pd
import os

def read_csv(filepath):
    return pd.read_csv(filepath)

def read_excel(filepath, sheet_name=0):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def read_text(filepath, delimiter="\t"):
    return pd.read_csv(filepath, delimiter=delimiter)

def save_to_csv(df, filepath, index=True):
    df.to_csv(filepath, index=index)

def save_to_excel(df, filepath, sheet_name='Sheet1', index=True):
    df.to_excel(filepath, sheet_name=sheet_name, index=index)

def get_sheetnames(fpath):
    """
    List all sheet names of an input Excel file (xlsx).
    
    Args:
    fpath (str): Path to the Excel file.
    
    Returns:
    list: A list of sheet names if the file exists and is an Excel file.
    None: If the file does not exist or is not an Excel file.
    """
    # Check if the file exists
    if not os.path.exists(fpath):
        print(f"The file {fpath} does not exist.")
        return None
    
    # Check if the file is an Excel file
    if not fpath.lower().endswith('.xlsx'):
        print(f"The file {fpath} is not an Excel file.")
        return None
    
    # Read the Excel file and get sheet names
    try:
        excel_file = pd.ExcelFile(fpath)
        sheet_names = excel_file.sheet_names
        return sheet_names
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return None
