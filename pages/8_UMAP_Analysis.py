"""
UMAP Analysis module — 2D & 3D dimensionality reduction with interactive controls.
Enhanced: Excel sheet support via load_data_widget, more metrics.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    download_plotly_html, download_dataframe, add_common_layout_options,
)

st.title("🗺️ UMAP Analysis")
st.markdown("---")

df_umap = load_data_widget("umap", "Upload data for UMAP")
if df_umap is not None:
    show_dataframe_overview(df_umap)

    num_cols = numeric_columns(df_umap)
    all_cols = df_umap.columns.tolist()

    # Column selection
    label_col = st.selectbox(
        "Label Column (optional, for coloring)",
        ["None"] + all_cols, key="umap_lbl",
    )
    label_col = None if label_col == "None" else label_col

    feature_cols = st.multiselect(
        "Feature Columns", num_cols, default=num_cols,
        key="umap_feat",
    )
    if len(feature_cols) < 2:
        st.warning("Select at least 2 numeric feature columns.")
        st.stop()

    # UMAP hyperparameters
    st.sidebar.header("⚙️ UMAP Settings")
    n_neighbors = st.sidebar.slider("Number of Neighbors", 2, 100, 15,
                                     key="umap_nn")
    min_dist = st.sidebar.slider("Minimum Distance", 0.0, 1.0, 0.1,
                                  0.01, key="umap_md")
    n_components = st.sidebar.radio("Components", [2, 3],
                                     key="umap_nc")
    metric = st.sidebar.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan", "cosine", "chebyshev", "correlation"],
        key="umap_metric",
    )
    random_state = int(st.sidebar.number_input("Random State",
                                                value=42, step=1,
                                                key="umap_rs"))

    if st.sidebar.button("🚀 Run UMAP", use_container_width=True,
                         key="umap_run"):
        with st.spinner("Running UMAP..."):
            X = df_umap[feature_cols].dropna()
            valid_idx = X.index
            X_scaled = StandardScaler().fit_transform(X)

            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=random_state,
            )
            embedding = reducer.fit_transform(X_scaled)

            cols = ([f"UMAP_{i+1}" for i in range(n_components)])
            umap_df = pd.DataFrame(embedding, columns=cols)

            color_col = None
            if label_col:
                umap_df["Label"] = df_umap.loc[valid_idx,
                                                label_col].values
                color_col = "Label"

            st.success("✅ UMAP completed!")

            if n_components == 2:
                fig = px.scatter(
                    umap_df, x="UMAP_1", y="UMAP_2",
                    color=color_col, opacity=0.7,
                    title="UMAP 2D Projection",
                )
                fig.update_traces(
                    marker=dict(size=8,
                                line=dict(width=0.5,
                                          color="DarkSlateGrey")))
            else:
                fig = px.scatter_3d(
                    umap_df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                    color=color_col, opacity=0.7,
                    title="UMAP 3D Projection",
                )
                fig.update_traces(marker=dict(size=5))

            fig = add_common_layout_options(fig, height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Parameters summary
            st.subheader("📝 Parameters Used")
            params = pd.DataFrame({
                "Parameter": ["n_neighbors", "min_dist",
                              "n_components", "metric",
                              "random_state", "n_features",
                              "n_samples"],
                "Value": [n_neighbors, min_dist, n_components,
                          metric, random_state,
                          len(feature_cols), len(valid_idx)],
            })
            st.table(params)

            download_plotly_html(fig, "umap_plot.html",
                                key="dl_umap_html")
            download_dataframe(umap_df, "umap_results.csv",
                               key="dl_umap_csv")
else:
    st.info("👈 Upload a CSV or Excel file to begin UMAP analysis.")