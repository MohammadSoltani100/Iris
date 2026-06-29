"""
Transcriptomics analysis page — parent page with three tabs:
  1) Transcriptomics Overview & Differential Expression Analysis
  2) Gene Expression Heatmap
  3) Volcano Plot
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    build_numeric_matrix, download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("🔬 Transcriptomics Analysis")
st.markdown("---")

tab_overview, tab_heatmap, tab_volcano = st.tabs([
    "📊 Transcriptomics Overview & DE",
    "🔥 Gene Expression Heatmap",
    "🌋 Volcano Plot",
])

# ═══════════════════════════════════════════
# TAB 1 — Transcriptomics Overview & DE
# ═══════════════════════════════════════════
with tab_overview:
    st.subheader("Transcriptomics Data Overview & Exploration")
    st.write("Upload a raw or normalized count matrix (rows = genes/transcripts, columns = samples).")

    df_t = load_data_widget("trans_over", "Upload expression count matrix")
    if df_t is not None:
        show_dataframe_overview(df_t)
        num_cols_t = numeric_columns(df_t)
        if len(num_cols_t) < 2:
            st.error("At least two numeric sample columns are required.")
            st.stop()

        gene_col_t = st.selectbox("Select Gene/Transcript ID Column", ["None"] + df_t.columns.tolist(), key="trans_gid")
        gene_col_t = None if gene_col_t == "None" else gene_col_t

        sample_cols_t = st.multiselect("Select Sample Columns", num_cols_t, default=num_cols_t[:min(12, len(num_cols_t))], key="trans_samples")
        if len(sample_cols_t) < 2:
            st.warning("Please select at least 2 sample columns.")
            st.stop()

        mat_t = build_numeric_matrix(df_t, gene_col_t, sample_cols_t, "Zero")

        # Optional log2 transformation
        log_transform = st.checkbox("Apply log2(x + 1) transformation for visualization", value=True, key="trans_log")
        plot_mat = np.log2(mat_t + 1) if log_transform else mat_t.copy()

        tc1, tc2 = st.columns(2)
        with tc1:
            # Sample expression distributions
            st.markdown("#### Sample Expression Distributions")
            melted_t = plot_mat.melt(var_name="Sample", value_name="Expression")
            fig_box = px.box(melted_t, x="Sample", y="Expression", color="Sample")
            fig_box = add_common_layout_options(fig_box, height=500, title="Sample Expression Distributions")
            st.plotly_chart(fig_box, use_container_width=True)

        with tc2:
            # PCA of Samples
            st.markdown("#### Sample Principal Component Analysis (PCA)")
            X_t = StandardScaler().fit_transform(plot_mat.T.values)
            pca_t = PCA(n_components=2)
            scores_t = pca_t.fit_transform(X_t)
            pca_df_t = pd.DataFrame(scores_t, columns=["PC1", "PC2"])
            pca_df_t["Sample"] = plot_mat.columns
            var_exp = pca_t.explained_variance_ratio_ * 100

            fig_pca_t = px.scatter(
                pca_df_t, x="PC1", y="PC2", text="Sample", color="Sample",
                labels={"PC1": f"PC1 ({var_exp[0]:.1f}%)", "PC2": f"PC2 ({var_exp[1]:.1f}%)"}
            )
            fig_pca_t.update_traces(textposition="top center")
            fig_pca_t = add_common_layout_options(fig_pca_t, height=500, title="Sample PCA")
            st.plotly_chart(fig_pca_t, use_container_width=True)

        # MA Plot / Pairwise Comparison
        st.markdown("#### Pairwise Sample Comparison (MA Plot / Scatter)")
        mc1, mc2 = st.columns(2)
        with mc1:
            s1 = st.selectbox("Sample 1 (Control / Base)", sample_cols_t, index=0, key="ma_s1")
        with mc2:
            s2 = st.selectbox("Sample 2 (Treatment / Comparison)", sample_cols_t, index=min(1, len(sample_cols_t)-1), key="ma_s2")

        ma_df = pd.DataFrame({
            "Gene": mat_t.index,
            "S1": mat_t[s1] + 1,
            "S2": mat_t[s2] + 1
        })
        ma_df["M (log2FC)"] = np.log2(ma_df["S2"] / ma_df["S1"])
        ma_df["A (Mean Log Exp)"] = 0.5 * (np.log2(ma_df["S2"]) + np.log2(ma_df["S1"]))

        fig_ma = px.scatter(ma_df, x="A (Mean Log Exp)", y="M (log2FC)", hover_data=["Gene"], opacity=0.6)
        fig_ma.add_hline(y=0, line_dash="dash", line_color="red")
        fig_ma = add_common_layout_options(fig_ma, height=550, title=f"MA Plot: {s2} vs {s1}")
        st.plotly_chart(fig_ma, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — Gene Expression Heatmap
# ═══════════════════════════════════════════
with tab_heatmap:
    st.subheader("Hierarchical Clustering Heatmap")
    st.write("Upload normalized gene expression data to generate a customized clustered heatmap.")

    df_h = load_data_widget("gene_heatmap", "Upload expression data for heatmap")
    if df_h is not None:
        show_dataframe_overview(df_h)
        num_cols_h = numeric_columns(df_h)
        if len(num_cols_h) < 2:
            st.error("At least two numeric expression columns required.")
            st.stop()

        id_col_h = st.selectbox("Gene ID column", ["None"] + df_h.columns.tolist(), key="heat_gid")
        id_col_h = None if id_col_h == "None" else id_col_h

        value_cols_h = st.multiselect("Expression columns", num_cols_h, default=num_cols_h[:min(12, len(num_cols_h))], key="heat_cols")
        if len(value_cols_h) < 2:
            st.warning("Select at least two columns.")
            st.stop()

        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            missing_method = st.selectbox("Missing values", ["Mean", "Median", "Zero", "Drop rows"], key="heat_miss")
        with hc2:
            scaling = st.selectbox("Scaling method", ["None", "Row Z-score", "Column Z-score"], key="heat_scale")
        with hc3:
            top_n = st.slider("Top most variable genes", 5, min(1000, len(df_h)), min(50, len(df_h)), key="heat_top")

        mat_h = build_numeric_matrix(df_h, id_col_h, value_cols_h, missing_method)

        if scaling == "Row Z-score":
            mat_h = mat_h.sub(mat_h.mean(axis=1), axis=0).div(mat_h.std(axis=1).replace(0, 1), axis=0)
        elif scaling == "Column Z-score":
            mat_h = mat_h.sub(mat_h.mean(axis=0), axis=1).div(mat_h.std(axis=0).replace(0, 1), axis=1)

        mat_h = mat_h.loc[mat_h.var(axis=1).sort_values(ascending=False).head(top_n).index]

        hc4, hc5, hc6 = st.columns(3)
        with hc4:
            cluster_rows = st.checkbox("Cluster rows (Genes)", True, key="heat_crows")
        with hc5:
            cluster_cols = st.checkbox("Cluster columns (Samples)", False, key="heat_ccols")
        with hc6:
            color_scale = st.selectbox("Color palette", ["RdBu_r", "Viridis", "Cividis", "Plasma", "Inferno"], key="heat_pal")

        plot_mat_h = mat_h.copy()
        if cluster_rows and plot_mat_h.shape[0] > 2:
            row_order = leaves_list(linkage(plot_mat_h.values, method="average"))
            plot_mat_h = plot_mat_h.iloc[row_order, :]

        if cluster_cols and plot_mat_h.shape[1] > 2:
            col_order = leaves_list(linkage(plot_mat_h.T.values, method="average"))
            plot_mat_h = plot_mat_h.iloc[:, col_order]

        fig_heat = px.imshow(
            plot_mat_h, color_continuous_scale=color_scale, aspect="auto",
            labels=dict(x="Samples", y="Genes", color="Expression")
        )
        fig_heat = add_common_layout_options(fig_heat, height=750, title="Gene Expression Clustered Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

        download_plotly_html(fig_heat, "gene_heatmap.html", key="dl_heat_html")
        download_dataframe(plot_mat_h.reset_index(), "heatmap_matrix.csv", key="dl_heat_csv")

# ═══════════════════════════════════════════
# TAB 3 — Volcano Plot
# ═══════════════════════════════════════════
with tab_volcano:
    st.subheader("Volcano Plot Analysis")
    st.write("Upload differential expression statistical results containing Gene ID, log2FoldChange, and P-value / FDR.")

    df_v = load_data_widget("volcano", "Upload differential expression table")
    if df_v is not None:
        show_dataframe_overview(df_v)
        num_cols_v = numeric_columns(df_v)
        if len(num_cols_v) < 2:
            st.error("At least two numeric columns (log2FC and p-value) are required.")
            st.stop()

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            gene_col_v = st.selectbox("Gene ID Column", df_v.columns.tolist(), key="vol_gid")
        with vc2:
            fc_col_v = st.selectbox("log2 Fold Change Column", num_cols_v, key="vol_fc")
        with vc3:
            p_col_v = st.selectbox("P-value / FDR Column", num_cols_v, key="vol_pval")

        df_vp = df_v.copy()
        df_vp[fc_col_v] = pd.to_numeric(df_vp[fc_col_v], errors="coerce")
        df_vp[p_col_v] = pd.to_numeric(df_vp[p_col_v], errors="coerce")
        df_vp = df_vp.replace([np.inf, -np.inf], np.nan).dropna(subset=[fc_col_v, p_col_v])
        df_vp = df_vp[df_vp[p_col_v] > 0]

        vc4, vc5, vc6 = st.columns(3)
        with vc4:
            fc_thresh = st.slider("Absolute log2FC Threshold", 0.0, 5.0, 1.0, 0.1, key="vol_fcth")
        with vc5:
            p_thresh = st.number_input("P-value Significance Threshold", min_value=1e-10, max_value=1.0, value=0.05, format="%.6f", key="vol_pth")
        with vc6:
            top_labels = st.slider("Number of Top Genes to Label", 0, 100, 20, key="vol_top")

        df_vp["neg_log10_p"] = -np.log10(df_vp[p_col_v])
        df_vp["Regulation"] = "Not Significant"
        df_vp.loc[(df_vp[p_col_v] < p_thresh) & (df_vp[fc_col_v] >= fc_thresh), "Regulation"] = "Up-regulated"
        df_vp.loc[(df_vp[p_col_v] < p_thresh) & (df_vp[fc_col_v] <= -fc_thresh), "Regulation"] = "Down-regulated"

        df_vp["Label"] = ""
        if top_labels > 0:
            sig_idx = df_vp[df_vp["Regulation"] != "Not Significant"].sort_values(p_col_v).head(top_labels).index
            df_vp.loc[sig_idx, "Label"] = df_vp.loc[sig_idx, gene_col_v].astype(str)

        colors_v = {"Up-regulated": "#D55E00", "Down-regulated": "#0072B2", "Not Significant": "#888888"}

        fig_vol = px.scatter(
            df_vp, x=fc_col_v, y="neg_log10_p", color="Regulation",
            color_discrete_map=colors_v, hover_data=[gene_col_v, fc_col_v, p_col_v],
            text="Label"
        )
        fig_vol.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="black")
        fig_vol.add_vline(x=fc_thresh, line_dash="dash", line_color="black")
        fig_vol.add_vline(x=-fc_thresh, line_dash="dash", line_color="black")
        fig_vol.update_traces(textposition="top center")
        fig_vol = add_common_layout_options(fig_vol, height=700, title="Volcano Plot")

        st.plotly_chart(fig_vol, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Up-regulated Genes", int((df_vp["Regulation"] == "Up-regulated").sum()))
        m2.metric("🔵 Down-regulated Genes", int((df_vp["Regulation"] == "Down-regulated").sum()))
        m3.metric("⚪ Not Significant", int((df_vp["Regulation"] == "Not Significant").sum()))

        download_plotly_html(fig_vol, "volcano_plot.html", key="dl_vol_html")
        download_dataframe(df_vp, "volcano_results.csv", key="dl_vol_csv")