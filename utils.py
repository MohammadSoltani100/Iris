"""
Utility helpers for the Multi-Omics Analysis Platform.
Backward-compatible with older page files.
NEW: Excel sheet selector built into load_data_widget.
"""

import io, os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─────────────────────────────────────────
# Internal readers
# ─────────────────────────────────────────
def _read_uploaded_table(uploaded_file, sheet_name=0):
    """Read CSV / TSV / TXT / XLSX / XLS and return DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def _get_sheet_names(uploaded_file):
    """Return list of sheet names for an Excel file."""
    try:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names
    except Exception:
        return []


# ─────────────────────────────────────────
# Main data loader with sheet selector
# ─────────────────────────────────────────
def load_and_validate_csv(uploaded_file, min_rows=2, min_cols=1, sheet_name=0):
    """Load and validate an uploaded file. Returns (df, error_msg)."""
    try:
        df = _read_uploaded_table(uploaded_file, sheet_name=sheet_name)
        if df is None:
            return None, "No file uploaded."
        if df.shape[0] < min_rows:
            return None, f"Need ≥{min_rows} rows (found {df.shape[0]})."
        if df.shape[1] < min_cols:
            return None, f"Need ≥{min_cols} columns (found {df.shape[1]})."
        return df, None
    except Exception as e:
        return None, f"Read error: {e}"


def load_data_widget(key_prefix, label="Upload data file",
                     min_rows=1, min_cols=1):
    """
    File uploader widget with **Excel sheet selector**.
    Returns pd.DataFrame or None.
    """
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        key=f"{key_prefix}_uploader",
    )
    if uploaded_file is None:
        return None

    # --- Sheet selector for Excel files ---
    sheet_name = 0
    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        sheets = _get_sheet_names(uploaded_file)
        if sheets and len(sheets) > 1:
            sheet_name = st.selectbox(
                "Select Excel sheet",
                sheets,
                key=f"{key_prefix}_sheet",
            )
        elif sheets:
            sheet_name = sheets[0]

    df, err = load_and_validate_csv(
        uploaded_file, min_rows=min_rows,
        min_cols=min_cols, sheet_name=sheet_name,
    )
    if err:
        st.error(err)
        return None
    return df


# ─────────────────────────────────────────
# Column helpers
# ─────────────────────────────────────────
def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def numeric_columns(df):
    return get_numeric_columns(df)

def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def categorical_columns(df):
    return get_categorical_columns(df)


# ─────────────────────────────────────────
# Data overview
# ─────────────────────────────────────────
def calculate_missing_stats(df):
    m = df.isnull().sum()
    mp = (m / len(df) * 100).round(2)
    s = pd.DataFrame({"Column": df.columns, "Missing": m.values,
                       "Missing%": mp.values, "Dtype": df.dtypes.values})
    return s[s["Missing"] > 0].sort_values("Missing", ascending=False)


def show_dataframe_overview(df, preview_rows=5):
    st.subheader("Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(get_numeric_columns(df)))
    c4.metric("Missing", int(df.isnull().sum().sum()))
    with st.expander("Preview", expanded=True):
        st.dataframe(df.head(preview_rows), use_container_width=True)
    ms = calculate_missing_stats(df)
    if not ms.empty:
        with st.expander("Missing values"):
            st.dataframe(ms, use_container_width=True)


# ─────────────────────────────────────────
# Numeric matrix builder
# ─────────────────────────────────────────
def build_numeric_matrix(df, id_col, columns, missing_method="Mean", **kw):
    """
    Build a numeric matrix. Accepts the legacy 4-arg call signature
    used by old page files: build_numeric_matrix(df, id_col, columns, missing_method).
    """
    if isinstance(columns, str):
        columns = [columns]
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return pd.DataFrame()

    mat = df[columns].copy()
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    if id_col is not None and id_col in df.columns:
        mat.index = df[id_col].astype(str)

    mm = missing_method.lower() if isinstance(missing_method, str) else "mean"
    if mm == "mean":
        mat = mat.fillna(mat.mean(numeric_only=True))
    elif mm == "median":
        mat = mat.fillna(mat.median(numeric_only=True))
    elif mm == "zero":
        mat = mat.fillna(0)
    else:
        mat = mat.dropna()

    return mat


# ─────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────
def download_plotly_html(fig, file_name="plot.html",
                         label="Download Plot (HTML)", key=None):
    if fig is None:
        return
    st.download_button(label, fig.to_html(include_plotlyjs="cdn", full_html=True),
                       file_name, "text/html", key=key)

def download_dataframe(df, file_name="data.csv",
                       label="Download Data (CSV)", index=False, key=None):
    if df is None:
        return
    st.download_button(label, df.to_csv(index=index),
                       file_name, "text/csv", key=key)


# ─────────────────────────────────────────
# Layout options
# ─────────────────────────────────────────
def add_common_layout_options(fig=None, key_prefix="layout",
                              height=600, **kwargs):
    """
    Simplified layout helper.
    - If fig is a Plotly figure: apply template + height and return fig.
    - If fig is a string: treat as key_prefix, return options dict.
    """
    if fig is not None and not hasattr(fig, "update_layout"):
        key_prefix = str(fig)
        fig = None

    if fig is None:
        return {"template": "plotly_white", "height": height}

    layout = {"template": "plotly_white", "height": height}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────
def standardize_features(X, return_scaler=False):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    return (Xs, sc) if return_scaler else Xs

def encode_labels(y):
    le = LabelEncoder(); ye = le.fit_transform(y)
    return ye, le, le.classes_.astype(str)

def safe_divide(a, b, default=0.0):
    try:
        return default if b == 0 else a / b
    except Exception:
        return default

def format_number(v, d=4):
    try:
        return f"{v:.{d}f}"
    except Exception:
        return str(v)