import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(page_title="Phenomics", layout="wide")

st.title("Phenomics analysis")
st.write("Upload phenotype data. Rows should represent genotypes, plots, samples, or observations.")

df = load_data_widget("phenomics", "Upload phenomics data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) == 0:
    st.error("No numeric phenotype traits were found.")
    st.stop()

group_col = st.selectbox("Group column for coloring or comparison", ["None"] + df.columns.tolist())
group_col = None if group_col == "None" else group_col

st.subheader("Correlation heatmap")
corr_cols = st.multiselect("Traits for correlation", num_cols, default=num_cols[: min(12, len(num_cols))])
corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])

if len(corr_cols) >= 2:
    corr = df[corr_cols].corr(method=corr_method)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    fig_corr = add_common_layout_options(fig_corr, "Trait correlation heatmap", height=650)
    st.plotly_chart(fig_corr, use_container_width=True)
    download_plotly_html(fig_corr, "phenomics_correlation_heatmap.html")

st.subheader("Trait visualization")
plot_type = st.selectbox("Plot type", ["Histogram + Box", "Box by group", "Scatter", "Scatter matrix"])

if plot_type == "Histogram + Box":
    trait = st.selectbox("Trait", num_cols)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box plot"])
    fig.add_trace(go.Histogram(x=df[trait], name="Histogram"), row=1, col=1)
    fig.add_trace(go.Box(y=df[trait], name="Box plot"), row=1, col=2)
    fig = add_common_layout_options(fig, f"Distribution of {trait}", height=500)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "phenomics_distribution.html")

elif plot_type == "Box by group":
    if group_col is None:
        st.warning("Select a group column first.")
    else:
        trait = st.selectbox("Trait", num_cols)
        fig = px.box(df, x=group_col, y=trait, color=group_col, points="all")
        fig = add_common_layout_options(fig, f"{trait} by {group_col}", height=600)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_html(fig, "phenomics_boxplot.html")

elif plot_type == "Scatter":
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X trait", num_cols)
    with c2:
        y_col = st.selectbox("Y trait", num_cols, index=min(1, len(num_cols) - 1))
    fig = px.scatter(df, x=x_col, y=y_col, color=group_col, hover_data=df.columns)
    fig = add_common_layout_options(fig, f"{y_col} vs {x_col}", height=650)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "phenomics_scatter.html")

else:
    traits = st.multiselect("Traits", num_cols, default=num_cols[: min(4, len(num_cols))])
    if len(traits) >= 2:
        fig = px.scatter_matrix(df, dimensions=traits, color=group_col)
        fig = add_common_layout_options(fig, "Trait scatter matrix", height=800)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_html(fig, "phenomics_scatter_matrix.html")

st.subheader("Descriptive statistics")
desc = df[num_cols].describe().T.reset_index().rename(columns={"index": "Trait"})
st.dataframe(desc, use_container_width=True)
download_dataframe(desc, "phenomics_descriptive_statistics.csv")