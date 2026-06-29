"""
PCA Analysis module — supports 2D/3D scatter, scree plot, loadings table.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    build_numeric_matrix, download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("📉 PCA Analysis")
st.markdown("---")
st.write("Upload a table where rows are samples and numeric columns are features.")

df_pca = load_data_widget("pca", "Upload PCA input data")
if df_pca is not None:
    show_dataframe_overview(df_pca)
    num_cols = numeric_columns(df_pca)
    if len(num_cols) < 2:
        st.error("PCA requires at least two numeric columns.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sample_col = st.selectbox("Sample ID Column",
                                  ["None"] + df_pca.columns.tolist(),
                                  key="pca_sid")
        sample_col = None if sample_col == "None" else sample_col
    with c2:
        group_col = st.selectbox("Group / Color Column",
                                 ["None"] + df_pca.columns.tolist(),
                                 key="pca_grp")
        group_col = None if group_col == "None" else group_col

    feature_cols = st.multiselect("Feature Columns for PCA", num_cols,
                                  default=num_cols, key="pca_feat")
    if len(feature_cols) < 2:
        st.warning("Select at least two feature columns.")
        st.stop()

    missing_method = st.selectbox("Missing Value Handling",
                                  ["Mean", "Median", "Zero", "Drop rows"],
                                  key="pca_miss")
    matrix = build_numeric_matrix(df_pca, sample_col, feature_cols,
                                  missing_method)

    scale_data = st.checkbox("Standardize Variables", True, key="pca_scale")
    X = matrix.values
    if scale_data:
        X = StandardScaler().fit_transform(X)

    max_comp = min(10, X.shape[0], X.shape[1])
    n_components = st.slider("Number of PCA Components", 2, max_comp,
                             min(5, max_comp), key="pca_ncomp")

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        scores,
        index=matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    pca_df["Sample"] = matrix.index.astype(str)

    # Map group labels to PCA df
    if group_col and sample_col:
        meta = (df_pca[[sample_col, group_col]]
                .drop_duplicates()
                .set_index(sample_col))
        pca_df[group_col] = (pca_df["Sample"]
                             .map(meta[group_col].astype(str)))
    elif group_col and not sample_col:
        if group_col in df_pca.columns:
            pca_df[group_col] = (df_pca.loc[matrix.index, group_col]
                                 .astype(str).values)

    var_exp = pca.explained_variance_ratio_ * 100

    plot_type = st.radio("PCA Plot Type", ["2D", "3D"],
                         horizontal=True, key="pca_dim")

    if plot_type == "2D":
        pc_names = [f"PC{i+1}" for i in range(n_components)]
        pc1, pc2 = st.columns(2)
        with pc1:
            x_pc = st.selectbox("X Component", pc_names, index=0,
                                key="pca_xpc")
        with pc2:
            y_pc = st.selectbox("Y Component", pc_names, index=1,
                                key="pca_ypc")

        x_idx = int(x_pc.replace("PC", "")) - 1
        y_idx = int(y_pc.replace("PC", "")) - 1

        show_labels = st.checkbox("Show Sample Labels", False,
                                  key="pca_labels")

        fig = px.scatter(
            pca_df, x=x_pc, y=y_pc,
            color=group_col if group_col else None,
            hover_data=["Sample"],
            text="Sample" if show_labels else None,
            labels={
                x_pc: f"{x_pc} ({var_exp[x_idx]:.1f}%)",
                y_pc: f"{y_pc} ({var_exp[y_idx]:.1f}%)",
            },
        )
        if show_labels:
            fig.update_traces(textposition="top center")
    else:
        fig = px.scatter_3d(
            pca_df, x="PC1", y="PC2",
            z="PC3" if n_components >= 3 else "PC2",
            color=group_col if group_col else None,
            hover_data=["Sample"],
            labels={
                "PC1": f"PC1 ({var_exp[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp[1]:.1f}%)",
                "PC3": f"PC3 ({var_exp[2]:.1f}%)"
                if n_components >= 3 else "",
            },
        )

    fig = add_common_layout_options(fig, height=700,
                                    title="PCA Score Plot")
    st.plotly_chart(fig, use_container_width=True)

    # Scree plot
    st.subheader("Scree Plot (Variance Explained)")
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Variance (%)": var_exp,
        "Cumulative (%)": np.cumsum(var_exp),
    })

    fig2 = px.bar(var_df, x="Component", y="Variance (%)",
                  text="Variance (%)", title="Scree Plot")
    fig2.update_traces(texttemplate="%{text:.1f}%",
                       textposition="outside")
    fig2.add_scatter(x=var_df["Component"],
                     y=var_df["Cumulative (%)"],
                     mode="lines+markers",
                     name="Cumulative %")
    fig2 = add_common_layout_options(fig2, height=450)
    st.plotly_chart(fig2, use_container_width=True)

    # Loadings table
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "Feature"})

    st.subheader("PCA Scores")
    st.dataframe(pca_df, use_container_width=True)
    st.subheader("PCA Loadings")
    st.dataframe(loadings, use_container_width=True)

    download_plotly_html(fig, "pca_plot.html", key="dl_pca_html")
    download_dataframe(pca_df.reset_index(drop=True),
                       "pca_scores.csv", key="dl_pca_scores")
    download_dataframe(loadings, "pca_loadings.csv",
                       key="dl_pca_load")