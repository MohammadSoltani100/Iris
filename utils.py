"""
Utility functions for the Multi-Omics Analysis Platform.
This module contains shared helper functions used across multiple pages.
The implementation is intentionally backward-compatible with older pages.
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =========================================================
# Internal file-reading helpers
# =========================================================
def _read_uploaded_table(uploaded_file):
    """
    Read an uploaded table file and return a pandas DataFrame.
    Supported formats: CSV, TSV, TXT, XLSX, XLS.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        pd.DataFrame
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    # Excel files
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # TSV files
    if file_name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")

    # TXT / CSV / unknown text-delimited files
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


# =========================================================
# General data-loading helpers
# =========================================================
def load_and_validate_csv(uploaded_file, min_rows=2, min_cols=1):
    """
    Load a file and perform basic validation.

    Args:
        uploaded_file: Streamlit uploaded file object
        min_rows: Minimum required number of rows
        min_cols: Minimum required number of columns

    Returns:
        tuple: (DataFrame, error_message)
    """
    try:
        df = _read_uploaded_table(uploaded_file)

        if df is None:
            return None, "No file was uploaded."

        if df.shape[0] < min_rows:
            return None, f"Dataset must have at least {min_rows} rows. Found {df.shape[0]}."

        if df.shape[1] < min_cols:
            return None, f"Dataset must have at least {min_cols} columns. Found {df.shape[1]}."

        return df, None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def load_data_widget(key_prefix, label="Upload data file", min_rows=1, min_cols=1):
    """
    Display a file uploader widget and return the uploaded DataFrame.

    This function is backward-compatible with older page files.

    Args:
        key_prefix: Unique key prefix for Streamlit widget
        label: File uploader label
        min_rows: Minimum required number of rows
        min_cols: Minimum required number of columns

    Returns:
        pd.DataFrame or None
    """
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        key=f"{key_prefix}_uploader"
    )

    if uploaded_file is None:
        return None

    df, error_message = load_and_validate_csv(
        uploaded_file,
        min_rows=min_rows,
        min_cols=min_cols
    )

    if error_message is not None:
        st.error(error_message)
        return None

    return df


# =========================================================
# Column helpers
# =========================================================
def get_numeric_columns(df):
    """
    Return numeric column names from a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        list
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def numeric_columns(df):
    """
    Backward-compatible alias for get_numeric_columns().
    """
    return get_numeric_columns(df)


def get_categorical_columns(df):
    """
    Return categorical column names from a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        list
    """
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def categorical_columns(df):
    """
    Backward-compatible alias for get_categorical_columns().
    """
    return get_categorical_columns(df)


# =========================================================
# Data overview helpers
# =========================================================
def calculate_missing_stats(df):
    """
    Calculate missing-value statistics for each column.

    Args:
        df: pandas DataFrame

    Returns:
        pandas DataFrame
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    stats_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
        "Data Type": df.dtypes.values
    })

    return stats_df[stats_df["Missing Count"] > 0].sort_values(
        "Missing Count",
        ascending=False
    )


def show_dataframe_overview(df, preview_rows=5):
    """
    Display a compact overview of a DataFrame in Streamlit.

    Args:
        df: pandas DataFrame
        preview_rows: Number of rows to show in preview
    """
    st.subheader("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(get_numeric_columns(df)))
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    with st.expander("Preview Data", expanded=True):
        st.dataframe(df.head(preview_rows), use_container_width=True)

    with st.expander("Column Types", expanded=False):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })
        st.dataframe(dtype_df, use_container_width=True)

    missing_df = calculate_missing_stats(df)
    if not missing_df.empty:
        with st.expander("Missing Value Summary", expanded=False):
            st.dataframe(missing_df, use_container_width=True)


# =========================================================
# Numeric matrix builder
# =========================================================
def build_numeric_matrix(
    df,
    columns=None,
    dropna=True,
    fillna_method=None,
    fill_value=None,
    scale=False,
    index_col=None,
    return_df=True,
    return_numpy=False,
    return_scaler=False,
    **kwargs
):
    """
    Build a clean numeric matrix from selected columns.

    This function is designed to be tolerant and backward-compatible
    with older page implementations.

    Args:
        df: pandas DataFrame
        columns: list of selected columns; if None, all numeric columns are used
        dropna: whether to drop rows containing NaN values
        fillna_method: one of ["mean", "median", "zero", "ffill", "bfill"]
        fill_value: custom fill value
        scale: whether to standardize features
        index_col: optional column to use as row index
        return_df: whether to return a DataFrame
        return_numpy: whether to return a NumPy array
        return_scaler: whether to return fitted scaler
        **kwargs: ignored extra arguments for compatibility

    Returns:
        DataFrame or ndarray, optionally with scaler
    """
    if columns is None:
        columns = get_numeric_columns(df)

    if isinstance(columns, str):
        columns = [columns]

    columns = [c for c in columns if c in df.columns]

    if len(columns) == 0:
        empty_df = pd.DataFrame()
        if return_scaler:
            return empty_df, None
        return empty_df

    matrix = df[columns].copy()

    for col in matrix.columns:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

    if index_col is not None and index_col in df.columns:
        matrix.index = df[index_col].astype(str)

    # Missing-value handling
    if fillna_method == "mean":
        matrix = matrix.fillna(matrix.mean(numeric_only=True))
    elif fillna_method == "median":
        matrix = matrix.fillna(matrix.median(numeric_only=True))
    elif fillna_method == "zero":
        matrix = matrix.fillna(0)
    elif fillna_method == "ffill":
        matrix = matrix.fillna(method="ffill")
    elif fillna_method == "bfill":
        matrix = matrix.fillna(method="bfill")
    elif fill_value is not None:
        matrix = matrix.fillna(fill_value)

    if dropna:
        matrix = matrix.dropna(axis=0)

    scaler = None
    if scale and len(matrix) > 0:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(matrix.values)
        matrix = pd.DataFrame(
            scaled_values,
            columns=matrix.columns,
            index=matrix.index
        )

    if return_numpy:
        result = matrix.values
    elif return_df:
        result = matrix
    else:
        result = matrix.values

    if return_scaler:
        return result, scaler

    return result


# =========================================================
# Plot / export helpers
# =========================================================
def download_plotly_html(fig, file_name="plot.html", label="Download Plot (HTML)", key=None):
    """
    Create a download button for a Plotly figure as HTML.

    Args:
        fig: Plotly figure
        file_name: output file name
        label: button label
        key: optional Streamlit key
    """
    if fig is None:
        st.warning("No plot is available for download.")
        return

    html_data = fig.to_html(include_plotlyjs="cdn", full_html=True)

    st.download_button(
        label=label,
        data=html_data,
        file_name=file_name,
        mime="text/html",
        key=key
    )


def download_dataframe(df, file_name="results.csv", label="Download Data (CSV)", index=False, key=None):
    """
    Create a download button for a DataFrame as CSV.

    Args:
        df: pandas DataFrame
        file_name: output file name
        label: button label
        index: whether to include index in CSV
        key: optional Streamlit key
    """
    if df is None:
        st.warning("No table is available for download.")
        return

    csv_data = df.to_csv(index=index)

    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        key=key
    )


# =========================================================
# Plot layout helper
# =========================================================
def add_common_layout_options(
    fig=None,
    key_prefix="layout",
    location="sidebar",
    default_template="plotly_white",
    default_height=600,
    default_width=None,
    default_showlegend=True,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    **kwargs
):
    """
    Add common Plotly layout options.

    This function supports two usage styles:

    1) options = add_common_layout_options("my_key")
       -> returns a dictionary of layout options from Streamlit widgets

    2) fig = add_common_layout_options(fig, key_prefix="my_key")
       -> applies layout settings directly to the figure and returns the figure

    Args:
        fig: Plotly figure or None
        key_prefix: unique widget key prefix
        location: "sidebar" or "main"
        default_template: default Plotly template
        default_height: default figure height
        default_width: default width (None = automatic)
        default_showlegend: default legend visibility
        title: optional plot title
        xaxis_title: optional x-axis title
        yaxis_title: optional y-axis title
        **kwargs: additional layout kwargs

    Returns:
        dict or plotly figure
    """
    # Backward-compatible mode:
    # if first positional argument is not a Plotly figure, treat it as key_prefix
    if fig is not None and not hasattr(fig, "update_layout"):
        key_prefix = str(fig)
        fig = None

    container = st.sidebar if location == "sidebar" else st

    with container.expander("Plot Layout Options", expanded=False):
        template = st.selectbox(
            "Template",
            ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            index=["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"].index(default_template)
            if default_template in ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
            else 0,
            key=f"{key_prefix}_template"
        )

        height = st.slider(
            "Figure Height",
            min_value=300,
            max_value=1200,
            value=default_height,
            step=50,
            key=f"{key_prefix}_height"
        )

        custom_width = st.checkbox(
            "Use Custom Width",
            value=default_width is not None,
            key=f"{key_prefix}_use_width"
        )

        width = None
        if custom_width:
            width_default = default_width if default_width is not None else 900
            width = st.number_input(
                "Figure Width",
                min_value=400,
                max_value=2000,
                value=int(width_default),
                step=50,
                key=f"{key_prefix}_width"
            )

        showlegend = st.checkbox(
            "Show Legend",
            value=default_showlegend,
            key=f"{key_prefix}_legend"
        )

    options = {
        "template": template,
        "height": height,
        "width": width,
        "showlegend": showlegend,
    }

    if fig is None:
        return options

    layout_kwargs = {
        "template": options["template"],
        "height": options["height"],
        "showlegend": options["showlegend"],
    }

    if options["width"] is not None:
        layout_kwargs["width"] = options["width"]

    if title is not None:
        layout_kwargs["title"] = title

    if xaxis_title is not None:
        layout_kwargs["xaxis_title"] = xaxis_title

    if yaxis_title is not None:
        layout_kwargs["yaxis_title"] = yaxis_title

    layout_kwargs.update(kwargs)
    fig.update_layout(**layout_kwargs)

    return fig


# =========================================================
# Miscellaneous helpers
# =========================================================
def standardize_features(X, return_scaler=False):
    """
    Standardize features using StandardScaler.

    Args:
        X: array-like or DataFrame
        return_scaler: whether to return scaler

    Returns:
        standardized array, optionally scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def encode_labels(y):
    """
    Encode categorical labels into numeric labels.

    Args:
        y: array-like labels

    Returns:
        tuple: (encoded_labels, label_encoder, class_names)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_.astype(str)
    return y_encoded, le, class_names


def safe_divide(a, b, default=0.0):
    """
    Safely divide two numbers.

    Args:
        a: numerator
        b: denominator
        default: fallback value if division fails

    Returns:
        float
    """
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def format_number(value, decimals=4):
    """
    Format a number with a fixed number of decimals.

    Args:
        value: input number
        decimals: number of decimal places

    Returns:
        str
    """
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)