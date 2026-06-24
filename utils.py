import io
import numpy as np
import pandas as pd
import streamlit as st


def load_data_widget(key: str, title: str = "Upload your data file"):
    uploaded = st.file_uploader(
        title,
        type=["csv", "txt", "tsv", "xlsx", "xls"],
        key=f"{key}_uploader"
    )

    if uploaded is None:
        st.info("Please upload a CSV, TXT, TSV, XLSX, or XLS file.")
        return None

    name = uploaded.name.lower()

    try:
        if name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox(
                "Select Excel sheet",
                xls.sheet_names,
                key=f"{key}_sheet"
            )
            df = pd.read_excel(xls, sheet_name=sheet)
        else:
            sep_choice = st.selectbox(
                "File separator",
                ["Auto", "Comma (,)", "Tab", "Semicolon (;)"],
                key=f"{key}_sep"
            )

            sep_map = {
                "Auto": None,
                "Comma (,)": ",",
                "Tab": "\t",
                "Semicolon (;)": ";"
            }

            df = pd.read_csv(
                uploaded,
                sep=sep_map[sep_choice],
                engine="python"
            )

        df.columns = [str(c).strip() for c in df.columns]
        return df

    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


def show_dataframe_overview(df: pd.DataFrame):
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


def coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def download_plotly_html(fig, filename: str):
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "Download plot as HTML",
        data=html,
        file_name=filename,
        mime="text/html"
    )


def download_dataframe(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def build_numeric_matrix(df: pd.DataFrame, id_col, value_cols, missing_method="Mean"):
    matrix = df[[id_col] + value_cols].copy() if id_col else df[value_cols].copy()

    if id_col:
        matrix[id_col] = matrix[id_col].astype(str)
        matrix = matrix.set_index(id_col)

    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if missing_method == "Mean":
        matrix = matrix.apply(lambda x: x.fillna(x.mean()), axis=0)
    elif missing_method == "Median":
        matrix = matrix.apply(lambda x: x.fillna(x.median()), axis=0)
    elif missing_method == "Zero":
        matrix = matrix.fillna(0)
    elif missing_method == "Drop rows":
        matrix = matrix.dropna(axis=0)

    return matrix


def add_common_layout_options(fig, title, height=650):
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    return fig