import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(page_title="Metabolomics", layout="wide")

st.title("Metabolomics analysis")
st.write("Upload metabolomics data in long format or wide format.")

df = load_data_widget("metabolomics", "Upload metabolomics data")

if df is None:
    st.stop()

show_dataframe_overview(df)

data_format = st.radio("Data format", ["Long format", "Wide format"], horizontal=True)

if data_format == "Long format":
    c1, c2, c3 = st.columns(3)
    with c1:
        metabolite_col = st.selectbox("Metabolite column", df.columns.tolist())
    with c2:
        value_col = st.selectbox("Concentration/value column", numeric_columns(df))
    with c3:
        group_col = st.selectbox("Treatment/group column", df.columns.tolist())

    selected_metabolites = st.multiselect(
        "Select metabolites",
        sorted(df[metabolite_col].dropna().astype(str).unique()),
        default=sorted(df[metabolite_col].dropna().astype(str).unique())[: min(8, df[metabolite_col].nunique())]
    )

    plot_df = df[df[metabolite_col].astype(str).isin(selected_metabolites)].copy()

    plot_type = st.selectbox("Plot type", ["Box plot", "Violin plot", "Mean bar plot", "Heatmap", "Radar plot"])

    if plot_type == "Box plot":
        fig = px.box(plot_df, x=metabolite_col, y=value_col, color=group_col, points="outliers")
    elif plot_type == "Violin plot":
        fig = px.violin(plot_df, x=metabolite_col, y=value_col, color=group_col, box=True, points="all")
    elif plot_type == "Mean bar plot":
        summary = plot_df.groupby([metabolite_col, group_col], as_index=False)[value_col].mean()
        fig = px.bar(summary, x=metabolite_col, y=value_col, color=group_col, barmode="group")
    elif plot_type == "Heatmap":
        pivot = plot_df.pivot_table(index=metabolite_col, columns=group_col, values=value_col, aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto")
    else:
        group = st.selectbox("Group for radar", sorted(plot_df[group_col].dropna().astype(str).unique()))
        radar = plot_df[plot_df[group_col].astype(str) == group].groupby(metabolite_col)[value_col].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar.values, theta=radar.index, fill="toself", name=group))

    fig = add_common_layout_options(fig, f"Metabolomics: {plot_type}", height=650)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "metabolomics_plot.html")
    download_dataframe(plot_df, "metabolomics_filtered_data.csv")

else:
    num_cols = numeric_columns(df)
    sample_col = st.selectbox("Sample ID column", ["None"] + df.columns.tolist())
    group_col = st.selectbox("Group column", ["None"] + df.columns.tolist())

    sample_col = None if sample_col == "None" else sample_col
    group_col = None if group_col == "None" else group_col

    metabolite_cols = st.multiselect("Metabolite numeric columns", num_cols, default=num_cols[: min(10, len(num_cols))])

    if len(metabolite_cols) < 2:
        st.warning("Select at least two metabolite columns.")
        st.stop()

    melted = df.melt(
        id_vars=[c for c in [sample_col, group_col] if c],
        value_vars=metabolite_cols,
        var_name="Metabolite",
        value_name="Concentration"
    )

    fig = px.box(
        melted,
        x="Metabolite",
        y="Concentration",
        color=group_col if group_col else None,
        points="outliers"
    )
    fig = add_common_layout_options(fig, "Metabolite profile from wide data", height=650)
    st.plotly_chart(fig, use_container_width=True)

    download_plotly_html(fig, "metabolomics_wide_boxplot.html")
    download_dataframe(melted, "metabolomics_long_converted.csv")