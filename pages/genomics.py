import numpy as np
import streamlit as st
import plotly.express as px

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(page_title="Genomics", layout="wide")

st.title("Genomics visualization")
st.write("Upload genomics results such as GWAS, marker statistics, gene annotations, or chromosome-position data.")

df = load_data_widget("genomics", "Upload genomics data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) == 0:
    st.error("At least one numeric column is required.")
    st.stop()

plot_type = st.selectbox(
    "Plot type",
    ["Manhattan plot", "Chromosome scatter", "Histogram", "Correlation heatmap"]
)

if plot_type in ["Manhattan plot", "Chromosome scatter"]:
    c1, c2, c3 = st.columns(3)
    with c1:
        chr_col = st.selectbox("Chromosome column", df.columns.tolist())
    with c2:
        pos_col = st.selectbox("Position column", num_cols)
    with c3:
        value_col = st.selectbox("Value column", num_cols)

    plot_df = df[[chr_col, pos_col, value_col]].copy()
    plot_df = plot_df.dropna()
    plot_df[chr_col] = plot_df[chr_col].astype(str)

    if plot_type == "Manhattan plot":
        transform_p = st.checkbox("Transform value as -log10(p)", True)
        if transform_p:
            plot_df["PlotValue"] = -np.log10(plot_df[value_col].clip(lower=1e-300))
            y_title = f"-log10({value_col})"
        else:
            plot_df["PlotValue"] = plot_df[value_col]
            y_title = value_col

        plot_df = plot_df.sort_values([chr_col, pos_col])
        fig = px.scatter(
            plot_df,
            x=pos_col,
            y="PlotValue",
            color=chr_col,
            hover_data=plot_df.columns,
            labels={"PlotValue": y_title}
        )
    else:
        fig = px.scatter(
            plot_df,
            x=pos_col,
            y=value_col,
            color=chr_col,
            hover_data=plot_df.columns
        )

elif plot_type == "Histogram":
    value_col = st.selectbox("Numeric column", num_cols)
    bins = st.slider("Bins", 5, 100, 40)
    fig = px.histogram(df, x=value_col, nbins=bins)

else:
    corr_cols = st.multiselect("Numeric columns", num_cols, default=num_cols[: min(12, len(num_cols))])
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")

fig = add_common_layout_options(fig, f"Genomics: {plot_type}", height=700)
st.plotly_chart(fig, use_container_width=True)

download_plotly_html(fig, "genomics_plot.html")
download_dataframe(df, "genomics_data.csv")