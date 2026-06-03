import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV or Excel file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception:
        # Fallback in case the user provides a real excel file instead of a CSV named .xls
        df = pd.read_excel(file_path)
        
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data by dropping duplicates and handling missing values.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Drop 'Id' column if it exists as it's not a useful feature
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    return df
