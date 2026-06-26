import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    build_numeric_matrix,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

#st.title("Cluster analysis")
st.write("Upload any numeric dataset and perform k-means or hierarchical clustering.")

df = load_data_widget("cluster", "Upload clustering data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) < 2:
    st.error("Clustering requires at least two numeric columns.")
    st.stop()

id_col = st.selectbox("Sample/Genotype ID column", ["None"] + df.columns.tolist())
id_col = None if id_col == "None" else id_col

features = st.multiselect("Features used for clustering", num_cols, default=num_cols[: min(20, len(num_cols))])

if len(features) < 2:
    st.warning("Select at least two features.")
    st.stop()

matrix = build_numeric_matrix(df, id_col, features, "Mean")

scale_data = st.checkbox("Standardize features", True)
X = matrix.values

if scale_data:
    X = StandardScaler().fit_transform(X)

method = st.selectbox("Clustering method", ["K-means", "Hierarchical"])
n_clusters = st.slider("Number of clusters", 2, min(12, len(matrix)), 3)

if method == "K-means":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X) + 1
else:
    linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    clusters = model.fit_predict(X) + 1

result = matrix.copy()
result["Cluster"] = clusters.astype(str)

st.subheader("Clustered samples")
st.dataframe(result.reset_index(), use_container_width=True)

x_feature = st.selectbox("X feature", features)
y_feature = st.selectbox("Y feature", features, index=min(1, len(features) - 1))

plot_df = result.reset_index()
id_name = plot_df.columns[0]

fig = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="Cluster",
    hover_data=[id_name],
    title="Cluster scatter plot"
)
fig = add_common_layout_options(fig, "Cluster scatter plot", height=650)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Dendrogram")
if len(matrix) <= 250:
    dendro_fig = ff.create_dendrogram(
        X,
        labels=matrix.index.astype(str).tolist(),
        orientation="left"
    )
    dendro_fig = add_common_layout_options(dendro_fig, "Hierarchical dendrogram", height=max(600, len(matrix) * 15))
    st.plotly_chart(dendro_fig, use_container_width=True)
    download_plotly_html(dendro_fig, "cluster_dendrogram.html")
else:
    st.warning("Dendrogram is disabled for more than 250 rows to keep the app responsive.")

download_plotly_html(fig, "cluster_scatter.html")
download_dataframe(result.reset_index(), "cluster_results.csv")