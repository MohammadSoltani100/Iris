"""
Cluster Analysis module — K-Means & Hierarchical clustering.
FIXED: Legend now displays real label names (not numeric cluster IDs).
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    build_numeric_matrix, download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("🔵 Cluster Analysis")
st.markdown("---")
st.write("Upload any numeric dataset to perform K-Means or Hierarchical clustering.")

df_cl = load_data_widget("cluster", "Upload clustering data")
if df_cl is not None:
    show_dataframe_overview(df_cl)
    num_cols = numeric_columns(df_cl)
    if len(num_cols) < 2:
        st.error("Clustering requires at least two numeric columns.")
        st.stop()

    # ── Column selection ──
    id_col = st.selectbox("Sample / Genotype ID Column",
                          ["None"] + df_cl.columns.tolist(),
                          key="cl_id")
    id_col = None if id_col == "None" else id_col

    # Optional: real label column for legend display
    label_col = st.selectbox(
        "Label Column for Legend (optional — shows real names instead of numbers)",
        ["None"] + df_cl.columns.tolist(),
        key="cl_label",
    )
    label_col = None if label_col == "None" else label_col

    features = st.multiselect(
        "Features for Clustering", num_cols,
        default=num_cols[:min(20, len(num_cols))],
        key="cl_feat",
    )
    if len(features) < 2:
        st.warning("Select at least two features.")
        st.stop()

    matrix = build_numeric_matrix(df_cl, id_col, features, "Mean")

    scale_data = st.checkbox("Standardize Features", True, key="cl_scale")
    X = matrix.values
    if scale_data:
        X = StandardScaler().fit_transform(X)

    # ── Clustering settings ──
    method = st.selectbox("Clustering Method",
                          ["K-means", "Hierarchical"], key="cl_method")
    n_clusters = st.slider("Number of Clusters", 2,
                           min(12, len(matrix)), 3, key="cl_k")

    if method == "K-means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
    else:
        linkage_method = st.selectbox("Linkage Method",
                                      ["ward", "complete", "average", "single"],
                                      key="cl_link")
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage=linkage_method)
        clusters = model.fit_predict(X)

    # ── Build result table ──
    result = matrix.copy()

    # Create descriptive cluster labels instead of plain numbers
    cluster_ids = clusters + 1  # 1-based
    result["Cluster_ID"] = cluster_ids

    # If user selected a label column, create descriptive cluster names
    # by finding the most common label in each cluster
    if label_col and label_col in df_cl.columns:
        # Map original labels to the matrix rows
        if id_col and id_col in df_cl.columns:
            label_map = (df_cl.set_index(id_col)[label_col]
                         .astype(str).to_dict())
            original_labels = [label_map.get(str(idx), str(idx))
                               for idx in matrix.index]
        else:
            original_labels = (df_cl.loc[matrix.index, label_col]
                               .astype(str).tolist()
                               if label_col in df_cl.columns
                               else [str(i) for i in matrix.index])

        result["Original_Label"] = original_labels

        # Build descriptive cluster names using most frequent label
        cluster_name_map = {}
        for cid in sorted(result["Cluster_ID"].unique()):
            mask = result["Cluster_ID"] == cid
            labels_in_cluster = result.loc[mask, "Original_Label"]
            most_common = labels_in_cluster.mode().iloc[0]
            count = mask.sum()
            cluster_name_map[cid] = f"Cluster {cid}: {most_common} (n={count})"

        result["Cluster"] = result["Cluster_ID"].map(cluster_name_map)
    else:
        result["Cluster"] = result["Cluster_ID"].apply(
            lambda x: f"Cluster {x}")

    # ── Silhouette score ──
    if len(set(clusters)) > 1 and len(X) > n_clusters:
        sil = silhouette_score(X, clusters)
        st.metric("Silhouette Score", f"{sil:.4f}",
                  help="Ranges from -1 to 1. Higher = better cluster separation.")

    # ── Clustered data table ──
    st.subheader("Clustered Samples")
    display_cols = (["Cluster"]
                    + (["Original_Label"] if label_col else [])
                    + features)
    st.dataframe(result[display_cols].reset_index(),
                 use_container_width=True)

    # ── Scatter plot ──
    st.subheader("Cluster Scatter Plot")
    sc1, sc2 = st.columns(2)
    with sc1:
        x_feat = st.selectbox("X Feature", features, index=0,
                               key="cl_xfeat")
    with sc2:
        y_feat = st.selectbox("Y Feature", features,
                               index=min(1, len(features)-1),
                               key="cl_yfeat")

    plot_df = result.reset_index()
    id_name = plot_df.columns[0]

    fig = px.scatter(
        plot_df, x=x_feat, y=y_feat,
        color="Cluster",
        hover_data=[id_name]
        + (["Original_Label"] if label_col else []),
        title="Cluster Scatter Plot",
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5,
                                                     color="DarkSlateGrey")))
    fig = add_common_layout_options(fig, height=650)
    st.plotly_chart(fig, use_container_width=True)

    # ── Dendrogram ──
    st.subheader("Dendrogram")
    if len(matrix) <= 250:
        dendro_labels = (result["Original_Label"].tolist()
                         if label_col
                         else matrix.index.astype(str).tolist())
        dendro_fig = ff.create_dendrogram(
            X,
            labels=dendro_labels,
            orientation="left",
        )
        dendro_fig.update_layout(
            height=max(600, len(matrix) * 18),
            template="plotly_white",
            title="Hierarchical Dendrogram",
        )
        st.plotly_chart(dendro_fig, use_container_width=True)
        download_plotly_html(dendro_fig, "cluster_dendrogram.html",
                             key="dl_cl_dendro")
    else:
        st.warning("Dendrogram disabled for > 250 samples.")

    # ── Cluster size bar chart ──
    st.subheader("Cluster Size Distribution")
    size_df = (result["Cluster"].value_counts()
               .reset_index()
               .rename(columns={"index": "Cluster",
                                "Cluster": "Count"})
               if pd.api.types.is_string_dtype(result["Cluster"])
               else result["Cluster"].value_counts().reset_index())
    size_df.columns = ["Cluster", "Count"]

    fig_size = px.bar(size_df, x="Cluster", y="Count",
                      color="Cluster", text="Count",
                      title="Samples per Cluster")
    fig_size.update_traces(textposition="outside")
    fig_size = add_common_layout_options(fig_size, height=450)
    st.plotly_chart(fig_size, use_container_width=True)

    # ── Downloads ──
    download_plotly_html(fig, "cluster_scatter.html",
                         key="dl_cl_scatter")
    download_dataframe(result.reset_index(),
                       "cluster_results.csv", key="dl_cl_csv")