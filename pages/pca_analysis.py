import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
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

st.set_page_config(page_title="PCA Analysis", layout="wide")

st.title("PCA analysis")
st.write("Upload a table where rows are samples and numeric columns are variables/features.")

df = load_data_widget("pca", "Upload PCA input data")

if df is None:
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if len(num_cols) < 2:
    st.error("PCA requires at least two numeric columns.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    sample_col = st.selectbox("Sample ID column", ["None"] + df.columns.tolist())
with c2:
    group_col = st.selectbox("Group/color column", ["None"] + df.columns.tolist())

sample_col = None if sample_col == "None" else sample_col
group_col = None if group_col == "None" else group_col

feature_cols = st.multiselect(
    "Feature columns used in PCA",
    num_cols,
    default=num_cols
)

if len(feature_cols) < 2:
    st.warning("Select at least two numeric feature columns.")
    st.stop()

missing_method = st.selectbox("Missing value handling", ["Mean", "Median", "Zero", "Drop rows"])
matrix = build_numeric_matrix(df, sample_col, feature_cols, missing_method)

scale_data = st.checkbox("Standardize variables", True)
X = matrix.values

if scale_data:
    X = StandardScaler().fit_transform(X)

max_comp = min(10, X.shape[0], X.shape[1])
n_components = st.slider("Number of PCA components", 2, max_comp, min(5, max_comp))

pca = PCA(n_components=n_components)
scores = pca.fit_transform(X)

pca_df = pd.DataFrame(scores, index=matrix.index, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["Sample"] = matrix.index.astype(str)

if group_col and sample_col:
    meta = df[[sample_col, group_col]].drop_duplicates().set_index(sample_col)
    pca_df[group_col] = pca_df["Sample"].map(meta[group_col].astype(str))
elif group_col and not sample_col:
    pca_df[group_col] = df.loc[matrix.index, group_col].astype(str).values if group_col in df.columns else None

plot_type = st.radio("PCA plot type", ["2D", "3D"], horizontal=True)

if plot_type == "2D":
    x_pc = st.selectbox("X component", pca_df.columns[:n_components], index=0)
    y_pc = st.selectbox("Y component", pca_df.columns[:n_components], index=1)
    fig = px.scatter(
        pca_df,
        x=x_pc,
        y=y_pc,
        color=group_col if group_col else None,
        hover_data=["Sample"],
        text="Sample" if st.checkbox("Show sample labels", False) else None
    )
else:
    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3" if n_components >= 3 else "PC2",
        color=group_col if group_col else None,
        hover_data=["Sample"]
    )

title = "PCA plot"
fig = add_common_layout_options(fig, title, height=700)
st.plotly_chart(fig, use_container_width=True)

variance_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_components)],
    "ExplainedVariance": pca.explained_variance_ratio_,
    "ExplainedVariancePercent": pca.explained_variance_ratio_ * 100
})

fig2 = px.bar(
    variance_df,
    x="Component",
    y="ExplainedVariancePercent",
    text="ExplainedVariancePercent",
    title="Scree plot"
)
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig2 = add_common_layout_options(fig2, "PCA explained variance", height=450)
st.plotly_chart(fig2, use_container_width=True)

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(n_components)]
).reset_index().rename(columns={"index": "Feature"})

st.subheader("PCA scores and loadings")
st.dataframe(pca_df, use_container_width=True)
st.dataframe(loadings, use_container_width=True)

download_plotly_html(fig, "pca_plot.html")
download_dataframe(pca_df.reset_index(drop=True), "pca_scores.csv")
download_dataframe(loadings, "pca_loadings.csv")