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

st.set_page_config(page_title="Volcano Plot", layout="wide")

st.title("Volcano plot")
st.write("Upload differential expression results with gene ID, log2 fold change, and p-value or adjusted p-value.")

df = load_data_widget("volcano", "Upload volcano plot data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) < 2:
    st.error("Volcano plot requires at least two numeric columns: log2FC and p-value.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    gene_col = st.selectbox("Gene ID column", df.columns.tolist())
with c2:
    fc_col = st.selectbox("log2 fold-change column", num_cols)
with c3:
    p_col = st.selectbox("p-value or adjusted p-value column", num_cols)

df_plot = df.copy()
df_plot[fc_col] = np.asarray(df_plot[fc_col], dtype=float)
df_plot[p_col] = np.asarray(df_plot[p_col], dtype=float)
df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=[fc_col, p_col])
df_plot = df_plot[df_plot[p_col] > 0]

c1, c2, c3 = st.columns(3)
with c1:
    fc_threshold = st.slider("Absolute log2FC threshold", 0.0, 5.0, 1.0, 0.1)
with c2:
    p_threshold = st.number_input("P-value threshold", min_value=0.000001, max_value=1.0, value=0.05, format="%.6f")
with c3:
    label_top_n = st.slider("Label top significant genes", 0, 100, 20)

df_plot["neg_log10_p"] = -np.log10(df_plot[p_col])

df_plot["Status"] = "Not significant"
df_plot.loc[(df_plot[p_col] < p_threshold) & (df_plot[fc_col] >= fc_threshold), "Status"] = "Up-regulated"
df_plot.loc[(df_plot[p_col] < p_threshold) & (df_plot[fc_col] <= -fc_threshold), "Status"] = "Down-regulated"

df_plot["Label"] = ""
if label_top_n > 0:
    top_idx = df_plot[df_plot["Status"] != "Not significant"].sort_values(p_col).head(label_top_n).index
    df_plot.loc[top_idx, "Label"] = df_plot.loc[top_idx, gene_col].astype(str)

colors = {
    "Up-regulated": "#D55E00",
    "Down-regulated": "#0072B2",
    "Not significant": "#777777"
}

fig = px.scatter(
    df_plot,
    x=fc_col,
    y="neg_log10_p",
    color="Status",
    color_discrete_map=colors,
    hover_data=[gene_col, fc_col, p_col],
    text="Label"
)

fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="black")
fig.add_vline(x=fc_threshold, line_dash="dash", line_color="black")
fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="black")
fig.update_traces(textposition="top center")
fig = add_common_layout_options(fig, "Volcano plot", height=700)

st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Up-regulated", int((df_plot["Status"] == "Up-regulated").sum()))
c2.metric("Down-regulated", int((df_plot["Status"] == "Down-regulated").sum()))
c3.metric("Not significant", int((df_plot["Status"] == "Not significant").sum()))

download_plotly_html(fig, "volcano_plot.html")
download_dataframe(df_plot, "volcano_results_classified.csv")