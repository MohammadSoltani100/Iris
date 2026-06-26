"""
Utility functions for the Multi-Omics Analysis Platform.
Shared helper functions used across multiple analysis pages.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_validate_csv(uploaded_file, min_rows=2, min_cols=1):
    """
    Load a CSV file and perform basic validation.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
    
    Returns:
        tuple: (DataFrame, error_message) - error_message is None if successful
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        if df.shape[0] < min_rows:
            return None, f"Dataset must have at least {min_rows} rows. Found {df.shape[0]}."
        
        if df.shape[1] < min_cols:
            return None, f"Dataset must have at least {min_cols} columns. Found {df.shape[1]}."
        
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"


def get_numeric_columns(df):
    """
    Get list of numeric columns from a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        list: List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """
    Get list of categorical columns from a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        list: List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def standardize_features(X, return_scaler=False):
    """
    Standardize features using StandardScaler.
    
    Args:
        X: numpy array or DataFrame of features
        return_scaler: If True, return the fitted scaler
    
    Returns:
        numpy array of standardized features, optionally the scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def encode_labels(y):
    """
    Encode categorical labels to numeric values.
    
    Args:
        y: array-like of labels
    
    Returns:
        tuple: (encoded_labels, label_encoder, class_names)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.astype(str)
    return y_encoded, le, class_names


def calculate_missing_stats(df):
    """
    Calculate missing value statistics for each column.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    stats_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values,
        'Data Type': df.dtypes.values
    })
    
    return stats_df[stats_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)


def safe_divide(a, b, default=0.0):
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero
    
    Returns:
        float: Result of division or default value
    """
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def format_number(value, decimals=4):
    """
    Format a number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        str: Formatted number string
    """
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)