import streamlit as st
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    build_numeric_matrix,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(page_title="Gene Heatmap", layout="wide")

st.title("Gene expression heatmap")
st.write("Upload a gene expression matrix. Recommended format: genes in rows and samples or conditions in columns.")

df = load_data_widget("gene_heatmap", "Upload gene expression data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) == 0:
    st.error("No numeric expression columns were found.")
    st.stop()

id_col = st.selectbox("Gene ID column", ["None"] + df.columns.tolist())
id_col = None if id_col == "None" else id_col

value_cols = st.multiselect(
    "Expression columns",
    num_cols,
    default=num_cols[: min(10, len(num_cols))]
)

if len(value_cols) < 2:
    st.warning("Select at least two numeric expression columns.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    missing_method = st.selectbox("Missing value handling", ["Mean", "Median", "Zero", "Drop rows"])
with c2:
    scaling = st.selectbox("Scaling", ["None", "Row Z-score", "Column Z-score"])
with c3:
    top_n = st.slider("Top variable genes/rows", 5, min(500, len(df)), min(50, len(df)))

matrix = build_numeric_matrix(df, id_col, value_cols, missing_method)

if scaling == "Row Z-score":
    matrix = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1).replace(0, 1), axis=0)
elif scaling == "Column Z-score":
    matrix = matrix.sub(matrix.mean(axis=0), axis=1).div(matrix.std(axis=0).replace(0, 1), axis=1)

matrix = matrix.loc[matrix.var(axis=1).sort_values(ascending=False).head(top_n).index]

st.sidebar.header("Heatmap options")
cluster_rows = st.sidebar.checkbox("Cluster rows", True)
cluster_cols = st.sidebar.checkbox("Cluster columns", False)
color_scale = st.sidebar.selectbox(
    "Color scale",
    ["RdBu_r", "Viridis", "Cividis", "Plasma", "Inferno", "YlGnBu", "RdYlGn"]
)
height = st.sidebar.slider("Plot height", 500, 1200, 750)

plot_matrix = matrix.copy()

if cluster_rows and plot_matrix.shape[0] > 2:
    row_order = leaves_list(linkage(plot_matrix.values, method="average"))
    plot_matrix = plot_matrix.iloc[row_order, :]

if cluster_cols and plot_matrix.shape[1] > 2:
    col_order = leaves_list(linkage(plot_matrix.T.values, method="average"))
    plot_matrix = plot_matrix.iloc[:, col_order]

fig = px.imshow(
    plot_matrix,
    color_continuous_scale=color_scale,
    aspect="auto",
    labels=dict(x="Samples / Conditions", y="Genes", color="Expression")
)
fig = add_common_layout_options(fig, "Gene expression heatmap", height=height)
st.plotly_chart(fig, use_container_width=True)

download_plotly_html(fig, "gene_expression_heatmap.html")
download_dataframe(plot_matrix.reset_index(), "heatmap_matrix.csv")