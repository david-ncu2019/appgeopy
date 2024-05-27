import pandas as pd

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