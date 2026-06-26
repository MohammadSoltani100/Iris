import numpy as np
import pandas as pd
import streamlit as st
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

#st.title("SNP analysis")
st.write("Upload SNP genotype matrix or GWAS results. Genotype matrix should preferably use dosage coding 0, 1, 2.")

df = load_data_widget("snp", "Upload SNP data")

if df is None:
    st.stop()

show_dataframe_overview(df)

analysis_type = st.radio("Analysis type", ["Genotype matrix", "GWAS table"], horizontal=True)

if analysis_type == "GWAS table":
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        st.error("GWAS table requires numeric position and p-value columns.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        chr_col = st.selectbox("Chromosome column", df.columns.tolist())
    with c2:
        pos_col = st.selectbox("Position column", num_cols)
    with c3:
        p_col = st.selectbox("P-value column", num_cols)

    plot_df = df[[chr_col, pos_col, p_col]].dropna().copy()
    plot_df[chr_col] = plot_df[chr_col].astype(str)
    plot_df["neg_log10_p"] = -np.log10(plot_df[p_col].clip(lower=1e-300))

    fig = px.scatter(
        plot_df,
        x=pos_col,
        y="neg_log10_p",
        color=chr_col,
        hover_data=plot_df.columns,
        labels={"neg_log10_p": f"-log10({p_col})"}
    )
    fig = add_common_layout_options(fig, "GWAS Manhattan-style plot", height=700)
    st.plotly_chart(fig, use_container_width=True)

    expected = -np.log10(np.linspace(1 / len(plot_df), 1, len(plot_df)))
    observed = -np.log10(np.sort(plot_df[p_col].clip(lower=1e-300)))
    qq = pd.DataFrame({"Expected": expected, "Observed": observed})

    fig2 = px.scatter(qq, x="Expected", y="Observed", title="QQ plot")
    fig2.add_shape(
        type="line",
        x0=qq["Expected"].min(),
        y0=qq["Expected"].min(),
        x1=qq["Expected"].max(),
        y1=qq["Expected"].max(),
        line=dict(color="black", dash="dash")
    )
    fig2 = add_common_layout_options(fig2, "GWAS QQ plot", height=600)
    st.plotly_chart(fig2, use_container_width=True)

    download_plotly_html(fig, "snp_gwas_manhattan.html")
    download_plotly_html(fig2, "snp_gwas_qq.html")

else:
    num_cols = numeric_columns(df)
    sample_col = st.selectbox("Sample ID column", ["None"] + df.columns.tolist())
    group_col = st.selectbox("Group column", ["None"] + df.columns.tolist())

    sample_col = None if sample_col == "None" else sample_col
    group_col = None if group_col == "None" else group_col

    snp_cols = st.multiselect("SNP marker columns", num_cols, default=num_cols[: min(200, len(num_cols))])

    if len(snp_cols) < 2:
        st.warning("Select at least two SNP marker columns.")
        st.stop()

    matrix = build_numeric_matrix(df, sample_col, snp_cols, "Mean")

    missing_rate_marker = df[snp_cols].isna().mean().reset_index()
    missing_rate_marker.columns = ["Marker", "MissingRate"]

    allele_p = matrix.mean(axis=0) / 2
    maf = allele_p.apply(lambda p: min(p, 1 - p)).reset_index()
    maf.columns = ["Marker", "MAF"]

    stats = missing_rate_marker.merge(maf, on="Marker")

    st.subheader("SNP quality summaries")
    c1, c2 = st.columns(2)

    fig_missing = px.histogram(stats, x="MissingRate", nbins=40, title="Marker missing rate")
    fig_maf = px.histogram(stats, x="MAF", nbins=40, title="Minor allele frequency")

    with c1:
        st.plotly_chart(add_common_layout_options(fig_missing, "Marker missing rate", height=450), use_container_width=True)
    with c2:
        st.plotly_chart(add_common_layout_options(fig_maf, "Minor allele frequency", height=450), use_container_width=True)

    st.subheader("SNP PCA")
    X = StandardScaler().fit_transform(matrix.values)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    pca_df = pd.DataFrame(scores, columns=["PC1", "PC2"])
    pca_df["Sample"] = matrix.index.astype(str)

    if group_col and sample_col:
        meta = df[[sample_col, group_col]].drop_duplicates().set_index(sample_col)
        pca_df[group_col] = pca_df["Sample"].map(meta[group_col].astype(str))

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=group_col if group_col else None,
        hover_data=["Sample"]
    )
    fig_pca = add_common_layout_options(fig_pca, "SNP PCA", height=650)
    st.plotly_chart(fig_pca, use_container_width=True)

    download_plotly_html(fig_pca, "snp_pca.html")
    download_dataframe(stats, "snp_marker_statistics.csv")