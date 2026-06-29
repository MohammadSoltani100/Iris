"""
Proteomics analysis page.
Comprehensive proteomics quantification analysis.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    build_numeric_matrix, download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("🧫 Proteomics Analysis")
st.markdown("---")
st.write(
    "Upload a protein quantification matrix "
    "(proteins × samples) — e.g. LFQ intensities, iBAQ, or normalized counts."
)

df_pr = load_data_widget("proteomics", "Upload proteomics data")
if df_pr is None:
    st.info("👈 Please upload a proteomics file from the panel above.")
    st.stop()

show_dataframe_overview(df_pr)
num_pr = numeric_columns(df_pr)
if len(num_pr) < 2:
    st.error("Need ≥ 2 numeric quantification columns.")
    st.stop()

# ─── Column selection ───
prot_col = st.selectbox(
    "Protein / accession ID column",
    ["None"] + df_pr.columns.tolist(), key="pr_pid")
prot_col = None if prot_col == "None" else prot_col

sample_cols = st.multiselect(
    "Sample (intensity) columns", num_pr,
    default=num_pr[:min(20, len(num_pr))], key="pr_samples")

if len(sample_cols) < 2:
    st.warning("Select ≥ 2 sample columns.")
    st.stop()

# ─── Group metadata (optional) ───
group_col = st.selectbox("Group / condition column (optional)",
                        ["None"] + df_pr.columns.tolist(),
                        key="pr_grp")
group_col = None if group_col == "None" else group_col

# ─── Normalization ───
st.sidebar.markdown("### Proteomics Options")
norm_pr = st.sidebar.selectbox(
    "Normalization",
    ["None", "log2(x+1)", "Z-score (rows)", "Median normalization"],
    key="pr_norm")

# ─── Build matrix ───
mat_pr = build_numeric_matrix(df_pr, prot_col, sample_cols, "Mean")

# Replace 0s with NaN for log
if norm_pr == "log2(x+1)":
    mat_pr = np.log2(mat_pr.clip(lower=0) + 1)
elif norm_pr == "Z-score (rows)":
    mat_pr = mat_pr.sub(mat_pr.mean(axis=1), axis=0).div(
        mat_pr.std(axis=1).replace(0, 1), axis=0)
elif norm_pr == "Median normalization":
    med = mat_pr.median(axis=0).replace(0, 1)
    mat_pr = mat_pr.div(med, axis=1) * med.median()

# ─── Analysis selector ───
analysis = st.selectbox(
    "Choose analysis",
    ["Intensity distribution",
     "Top abundant proteins",
     "Missing value pattern",
     "Sample correlation",
     "Sample PCA",
     "Coefficient of variation (CV)",
     "Differential abundance (t-test)",
     "Heatmap"],
    key="pr_an")

# ════════════════════════════════════════════
# 1) Intensity distribution
# ════════════════════════════════════════════
if analysis == "Intensity distribution":
    melt = mat_pr.reset_index().melt(
        id_vars=mat_pr.index.name or "index",
        var_name="Sample", value_name="Intensity")
    fig = px.box(melt, x="Sample", y="Intensity",
                 title="Protein Intensity Distribution per Sample")
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "proteomics_intensity.html", key="dl_pr_int")

# ════════════════════════════════════════════
# 2) Top abundant proteins
# ════════════════════════════════════════════
elif analysis == "Top abundant proteins":
    top_n_pr = st.slider("Top N proteins", 5, 100, 25, key="pr_topn")
    means = mat_pr.mean(axis=1).sort_values(ascending=False).head(top_n_pr)
    fig = px.bar(x=means.index.astype(str), y=means.values,
                 labels={"x": "Protein", "y": "Mean Intensity"},
                 title=f"Top {top_n_pr} Most Abundant Proteins")
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# 3) Missing value pattern
# ════════════════════════════════════════════
elif analysis == "Missing value pattern":
    miss_per_sample = mat_pr.isna().sum(axis=0)
    miss_per_protein = mat_pr.isna().sum(axis=1)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(x=miss_per_sample.index,
                      y=miss_per_sample.values,
                      labels={"x": "Sample", "y": "Missing count"},
                      title="Missing Values per Sample")
        fig1.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(x=miss_per_protein.values, nbins=30,
                            labels={"x": "Missing count per protein"},
                            title="Missing-Value Distribution (Proteins)")
        fig2.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════
# 4) Sample correlation
# ════════════════════════════════════════════
elif analysis == "Sample correlation":
    cm = st.selectbox("Correlation method",
                      ["pearson", "spearman"], key="pr_cm")
    corr = mat_pr.corr(method=cm)
    fig = px.imshow(corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title=f"Sample-Sample Correlation ({cm})")
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# 5) Sample PCA
# ════════════════════════════════════════════
elif analysis == "Sample PCA":
    fill_mat = mat_pr.fillna(mat_pr.mean(axis=1).values[:, None])
    Xpr = StandardScaler().fit_transform(fill_mat.T.values)
    pca_pr = PCA(n_components=2)
    scrs = pca_pr.fit_transform(Xpr)
    pdf = pd.DataFrame(scrs, columns=["PC1", "PC2"])
    pdf["Sample"] = mat_pr.columns

    color = None
    if group_col and group_col in df_pr.columns:
        # Build sample -> group lookup
        gmap = dict(zip(df_pr.columns, [group_col]))
        try:
            # Heuristic: if group_col equals a row in the table mapping
            # samples to groups, attempt a join. Fallback: ignore.
            mapping = {}
            for s in pdf["Sample"]:
                if s in df_pr[group_col].astype(str).values:
                    mapping[s] = s
            pdf[group_col] = pdf["Sample"].map(mapping)
            if pdf[group_col].notna().any():
                color = group_col
        except Exception:
            pass

    fig = px.scatter(pdf, x="PC1", y="PC2", text="Sample",
                     color=color,
                     title=(f"Sample PCA — PC1 "
                            f"({pca_pr.explained_variance_ratio_[0]*100:.1f} %)"
                            f" / PC2 "
                            f"({pca_pr.explained_variance_ratio_[1]*100:.1f} %)"))
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# 6) CV analysis
# ════════════════════════════════════════════
elif analysis == "Coefficient of variation (CV)":
    means = mat_pr.mean(axis=1)
    stds = mat_pr.std(axis=1)
    cv = (stds / means.replace(0, np.nan)) * 100
    cv_df = pd.DataFrame({"Protein": mat_pr.index.astype(str),
                          "Mean": means.values,
                          "SD": stds.values,
                          "CV%": cv.values}).dropna()

    fig = px.histogram(cv_df, x="CV%", nbins=40,
                       title="Coefficient of Variation Distribution")
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(cv_df.sort_values("CV%", ascending=False),
                 use_container_width=True)
    download_dataframe(cv_df, "proteomics_cv.csv", key="dl_pr_cv")

# ════════════════════════════════════════════
# 7) Differential abundance (t-test)
# ════════════════════════════════════════════
elif analysis == "Differential abundance (t-test)":
    st.markdown("Provide two **disjoint** sample groups:")
    g1 = st.multiselect("Group A samples", sample_cols, key="pr_g1")
    g2 = st.multiselect(
        "Group B samples",
        [s for s in sample_cols if s not in g1],
        key="pr_g2")

    if len(g1) >= 2 and len(g2) >= 2:
        from scipy.stats import ttest_ind
        with np.errstate(invalid="ignore"):
            t_stat, p_val = ttest_ind(
                mat_pr[g1].values, mat_pr[g2].values,
                axis=1, equal_var=False, nan_policy="omit")
        log2fc = (mat_pr[g1].mean(axis=1)
                  - mat_pr[g2].mean(axis=1))

        de_df = pd.DataFrame({
            "Protein": mat_pr.index.astype(str),
            "MeanA": mat_pr[g1].mean(axis=1).values,
            "MeanB": mat_pr[g2].mean(axis=1).values,
            "log2FC(A-B)": log2fc.values,
            "t_stat": t_stat,
            "p_value": p_val,
        }).dropna()
        de_df["neg_log10_p"] = -np.log10(de_df["p_value"].clip(lower=1e-300))

        p_th_pr = st.number_input("Significance p-value", value=0.05,
                                  min_value=0.000001, max_value=1.0,
                                  format="%.6f", key="pr_pth")
        fc_th_pr = st.slider("|log2FC| threshold", 0.0, 5.0, 1.0, 0.1,
                             key="pr_fcth")

        de_df["Status"] = "NS"
        de_df.loc[(de_df["p_value"] < p_th_pr) &
                   (de_df["log2FC(A-B)"] >= fc_th_pr), "Status"] = "Up"
        de_df.loc[(de_df["p_value"] < p_th_pr) &
                   (de_df["log2FC(A-B)"] <= -fc_th_pr), "Status"] = "Down"

        fig = px.scatter(de_df, x="log2FC(A-B)", y="neg_log10_p",
                         color="Status",
                         color_discrete_map={"Up": "#D55E00",
                                              "Down": "#0072B2",
                                              "NS": "#888"},
                         hover_data=["Protein"],
                         title="Proteomics Volcano Plot")
        fig.add_hline(y=-np.log10(p_th_pr),
                      line_dash="dash", line_color="black")
        fig.add_vline(x=fc_th_pr, line_dash="dash", line_color="black")
        fig.add_vline(x=-fc_th_pr, line_dash="dash", line_color="black")
        fig.update_layout(height=650, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(de_df.sort_values("p_value"),
                     use_container_width=True)
        download_dataframe(de_df, "proteomics_DE.csv", key="dl_pr_de")
    else:
        st.info("Select at least 2 samples in each group.")

# ════════════════════════════════════════════
# 8) Heatmap
# ════════════════════════════════════════════
elif analysis == "Heatmap":
    top_n_h = st.slider("Top N variable proteins",
                        5, min(500, len(mat_pr)),
                        min(50, len(mat_pr)), key="pr_h_top")
    var_top = mat_pr.var(axis=1).sort_values(
        ascending=False).head(top_n_h).index
    mat_show = mat_pr.loc[var_top]

    if mat_show.shape[0] > 2:
        ro = leaves_list(linkage(mat_show.fillna(0).values,
                                  method="average"))
        mat_show = mat_show.iloc[ro, :]

    fig = px.imshow(mat_show, color_continuous_scale="RdBu_r",
                    aspect="auto",
                    labels=dict(x="Sample", y="Protein",
                                color="Intensity"),
                    title=f"Proteomics Heatmap (top {top_n_h})")
    fig.update_layout(height=750, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    download_plotly_html(fig, "proteomics_heatmap.html",
                         key="dl_pr_hmap")
    download_dataframe(mat_show.reset_index(),
                       "proteomics_heatmap_matrix.csv",
                       key="dl_pr_hmap_csv")