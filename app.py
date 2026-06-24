import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    categorical_columns,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(
    page_title="BioData Visualization App",
    page_icon="chart",
    layout="wide"
)

st.title("BioData Visualization App")
st.write(
    "Upload your own CSV or Excel file and create customizable scientific plots. "
    "Use the pages in the left sidebar for gene expression, PCA, volcano plot, genomics, SNP, phenomics, metabolomics, pathway, and clustering analyses."
)

df = load_data_widget("main", "Upload a general data file")

if df is not None:
    show_dataframe_overview(df)

    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

    if len(num_cols) == 0:
        st.error("Your file must contain at least one numeric column.")
        st.stop()

    st.subheader("General plotting")

    chart_type = st.selectbox(
        "Chart type",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Correlation Heatmap"]
    )

    color_col = st.selectbox(
        "Color/group column",
        ["None"] + df.columns.tolist()
    )

    if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot"]:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X-axis", df.columns.tolist())
        with c2:
            y_col = st.selectbox("Y-axis", num_cols)

    title = st.text_input("Plot title", value=chart_type)
    color_arg = None if color_col == "None" else color_col

    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, hover_data=df.columns)
    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col, color=color_arg)
    elif chart_type == "Bar Chart":
        agg_method = st.selectbox("Aggregation", ["None", "Mean", "Sum", "Median"])
        if agg_method != "None":
            agg_fun = agg_method.lower()
            plot_df = df.groupby(x_col, as_index=False)[y_col].agg(agg_fun)
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_arg if color_arg in plot_df.columns else None)
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_arg)
    elif chart_type == "Histogram":
        hist_col = st.selectbox("Numeric variable", num_cols)
        bins = st.slider("Number of bins", 5, 100, 30)
        fig = px.histogram(df, x=hist_col, color=color_arg, nbins=bins)
    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col, color=color_arg, points="outliers")
    else:
        corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        corr = df[num_cols].corr(method=corr_method)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

    fig = add_common_layout_options(fig, title)
    st.plotly_chart(fig, use_container_width=True)

    download_plotly_html(fig, "general_plot.html")
    download_dataframe(df, "uploaded_data.csv")

else:
    st.info("Upload a dataset to start.")
    st.markdown(
        """
        Suggested formats:

        - General plots: any table with numeric columns
        - Heatmap/PCA: samples in rows and numeric variables in columns
        - Volcano plot: gene ID, log2 fold change, p-value or adjusted p-value
        - SNP analysis: marker columns coded as 0/1/2, plus optional sample metadata
        - Pathway: edge list with source and target columns
        """
    )