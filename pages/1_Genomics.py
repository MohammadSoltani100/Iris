"""
Genomics analysis page — parent page with three tabs:
  1) Genomics Analysis (GWAS Manhattan + QQ + general)
  2) SNP Analysis
  3) Phylogenetic Analysis
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    build_numeric_matrix, download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("🧬 Genomics Analysis")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Three tabs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_gen, tab_snp, tab_phylo = st.tabs([
    "🧬 Genomics / GWAS",
    "🔬 SNP Analysis",
    "🌳 Phylogenetic Analysis",
])

# ═══════════════════════════════════════════
# TAB 1 — Genomics / GWAS
# ═══════════════════════════════════════════
with tab_gen:
    st.subheader("Genomics & GWAS Visualization")
    st.write("Upload GWAS results or any chromosome-position-value data.")

    df_g = load_data_widget("genomics", "Upload genomics / GWAS data")
    if df_g is not None:
        show_dataframe_overview(df_g)
        num_cols_g = numeric_columns(df_g)
        if len(num_cols_g) == 0:
            st.error("At least one numeric column is required.")
            st.stop()

        plot_type = st.selectbox(
            "Plot type",
            ["Manhattan plot", "QQ plot", "Chromosome scatter",
             "Histogram", "Correlation heatmap"],
            key="gen_plot_type",
        )

        # ── Manhattan & QQ shared columns ──
        if plot_type in ["Manhattan plot", "QQ plot", "Chromosome scatter"]:
            c1, c2, c3 = st.columns(3)
            with c1:
                chr_col = st.selectbox("Chromosome column",
                                       df_g.columns.tolist(), key="gen_chr")
            with c2:
                pos_col = st.selectbox("Position column",
                                       num_cols_g, key="gen_pos")
            with c3:
                p_col = st.selectbox("P-value column",
                                     num_cols_g, key="gen_pval")

            plot_df = df_g[[chr_col, pos_col, p_col]].dropna().copy()
            plot_df[chr_col] = plot_df[chr_col].astype(str)

        # ── Manhattan plot (improved) ──
        if plot_type == "Manhattan plot":
            st.markdown("#### Manhattan Plot Settings")
            mc1, mc2 = st.columns(2)
            with mc1:
                sig_level = st.number_input(
                    "Genome-wide significance (-log10 p)",
                    value=7.3, min_value=0.0, max_value=20.0,
                    step=0.1, key="gen_sig",
                )
            with mc2:
                sug_level = st.number_input(
                    "Suggestive threshold (-log10 p)",
                    value=5.0, min_value=0.0, max_value=20.0,
                    step=0.1, key="gen_sug",
                )

            plot_df["neg_log10_p"] = -np.log10(
                plot_df[p_col].clip(lower=1e-300))

            # Build cumulative genome position for proper Manhattan
            chr_order = sorted(plot_df[chr_col].unique(),
                               key=lambda x: (
                                   int(x) if x.isdigit() else 999, x))
            plot_df[chr_col] = pd.Categorical(
                plot_df[chr_col], categories=chr_order, ordered=True)
            plot_df = plot_df.sort_values([chr_col, pos_col])

            chr_offset = {}
            cumulative = 0
            chr_centers = {}
            for ch in chr_order:
                chr_offset[ch] = cumulative
                ch_data = plot_df[plot_df[chr_col] == ch]
                ch_max = ch_data[pos_col].max() if len(ch_data) > 0 else 0
                chr_centers[ch] = cumulative + ch_max / 2
                cumulative += ch_max

            plot_df["genome_pos"] = plot_df.apply(
                lambda r: r[pos_col] + chr_offset[r[chr_col]], axis=1)

            # Alternating colors
            color_list = ["#1f77b4", "#ff7f0e"] * ((len(chr_order) // 2) + 1)
            color_map = {ch: color_list[i] for i, ch in enumerate(chr_order)}
            plot_df["chr_color"] = plot_df[chr_col].map(color_map)

            fig = go.Figure()
            for ch in chr_order:
                ch_df = plot_df[plot_df[chr_col] == ch]
                fig.add_trace(go.Scattergl(
                    x=ch_df["genome_pos"], y=ch_df["neg_log10_p"],
                    mode="markers",
                    marker=dict(size=4, color=color_map[ch]),
                    name=f"Chr {ch}",
                    text=ch_df.index,
                    hovertemplate=(
                        f"Chr {ch}<br>"
                        f"Pos: %{{customdata[0]:,.0f}}<br>"
                        f"-log10(p): %{{y:.2f}}<extra></extra>"
                    ),
                    customdata=ch_df[[pos_col]].values,
                ))

            # Threshold lines
            fig.add_hline(y=sig_level, line_dash="dash",
                          line_color="red", line_width=1.5,
                          annotation_text=f"Significance ({sig_level})",
                          annotation_position="top left")
            fig.add_hline(y=sug_level, line_dash="dot",
                          line_color="blue", line_width=1,
                          annotation_text=f"Suggestive ({sug_level})",
                          annotation_position="top left")

            fig.update_layout(
                xaxis=dict(
                    tickvals=list(chr_centers.values()),
                    ticktext=list(chr_centers.keys()),
                    title="Chromosome",
                ),
                yaxis_title=f"-log₁₀({p_col})",
                template="plotly_white",
                height=600,
                showlegend=False,
                title="Manhattan Plot",
            )
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_html(fig, "manhattan_plot.html",
                                key="dl_manhattan")

        # ── QQ plot (improved) ──
        elif plot_type == "QQ plot":
            pvals = plot_df[p_col].clip(lower=1e-300).sort_values().values
            n = len(pvals)
            expected = -np.log10(np.arange(1, n + 1) / (n + 1))
            observed = -np.log10(pvals)

            qq = pd.DataFrame({"Expected": sorted(expected),
                               "Observed": sorted(observed, reverse=True)})

            # Confidence interval (95 %)
            from scipy.stats import beta as beta_dist
            ci_lo = -np.log10(
                beta_dist.ppf(0.975, np.arange(1, n+1), np.arange(n, 0, -1)))
            ci_hi = -np.log10(
                beta_dist.ppf(0.025, np.arange(1, n+1), np.arange(n, 0, -1)))

            fig_qq = go.Figure()
            # Confidence band
            exp_sorted = sorted(expected)
            fig_qq.add_trace(go.Scatter(
                x=exp_sorted, y=ci_hi[np.argsort(expected)],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig_qq.add_trace(go.Scatter(
                x=exp_sorted, y=ci_lo[np.argsort(expected)],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(200,200,200,0.4)",
                name="95% CI",
            ))
            # Diagonal
            mx = max(qq["Expected"].max(), qq["Observed"].max())
            fig_qq.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines",
                line=dict(color="red", dash="dash"), name="Expected",
            ))
            # Points
            fig_qq.add_trace(go.Scatter(
                x=qq["Expected"], y=qq["Observed"],
                mode="markers",
                marker=dict(size=4, color="#1f77b4"),
                name="Observed",
            ))
            fig_qq.update_layout(
                xaxis_title="Expected -log₁₀(p)",
                yaxis_title="Observed -log₁₀(p)",
                title="QQ Plot", template="plotly_white", height=600,
            )
            st.plotly_chart(fig_qq, use_container_width=True)
            download_plotly_html(fig_qq, "qq_plot.html", key="dl_qq")

        # ── Chromosome scatter ──
        elif plot_type == "Chromosome scatter":
            fig = px.scatter(plot_df, x=pos_col, y=p_col,
                             color=chr_col, hover_data=plot_df.columns)
            fig = add_common_layout_options(fig, height=650)
            st.plotly_chart(fig, use_container_width=True)

        # ── Histogram ──
        elif plot_type == "Histogram":
            val_col = st.selectbox("Column", num_cols_g, key="gen_hist")
            bins = st.slider("Bins", 5, 100, 40, key="gen_bins")
            fig = px.histogram(df_g, x=val_col, nbins=bins)
            fig = add_common_layout_options(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation heatmap ──
        else:
            cc = st.multiselect("Columns", num_cols_g,
                                default=num_cols_g[:min(12, len(num_cols_g))],
                                key="gen_corr")
            if len(cc) >= 2:
                corr = df_g[cc].corr()
                fig = px.imshow(corr, text_auto=".2f",
                                color_continuous_scale="RdBu_r",
                                aspect="auto")
                fig = add_common_layout_options(fig, height=650)
                st.plotly_chart(fig, use_container_width=True)

        download_dataframe(df_g, "genomics_data.csv", key="dl_gen_data")

# ═══════════════════════════════════════════
# TAB 2 — SNP Analysis
# ═══════════════════════════════════════════
with tab_snp:
    st.subheader("SNP Analysis")
    st.write("Upload SNP genotype matrix (0/1/2 dosage) or GWAS summary table.")

    df_s = load_data_widget("snp", "Upload SNP data")
    if df_s is not None:
        show_dataframe_overview(df_s)
        analysis_type = st.radio("Analysis type",
                                 ["Genotype matrix", "GWAS table"],
                                 horizontal=True, key="snp_atype")

        if analysis_type == "GWAS table":
            num_s = numeric_columns(df_s)
            if len(num_s) < 2:
                st.error("Need numeric position and p-value columns.")
                st.stop()

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                chr_col_s = st.selectbox("Chromosome", df_s.columns.tolist(),
                                         key="snp_chr")
            with sc2:
                pos_col_s = st.selectbox("Position", num_s, key="snp_pos")
            with sc3:
                p_col_s = st.selectbox("P-value", num_s, key="snp_pval")

            # Manhattan
            gdf = df_s[[chr_col_s, pos_col_s, p_col_s]].dropna().copy()
            gdf[chr_col_s] = gdf[chr_col_s].astype(str)
            gdf["neg_log10_p"] = -np.log10(gdf[p_col_s].clip(lower=1e-300))

            threshold_snp = st.number_input(
                "Significance threshold (-log10 p)", value=5.0,
                key="snp_thresh")

            fig_m = px.scatter(gdf, x=pos_col_s, y="neg_log10_p",
                               color=chr_col_s, hover_data=gdf.columns)
            fig_m.add_hline(y=threshold_snp, line_dash="dash",
                            line_color="red")
            fig_m = add_common_layout_options(fig_m, height=650)
            st.plotly_chart(fig_m, use_container_width=True)

            # QQ plot
            pvals_s = gdf[p_col_s].clip(lower=1e-300).sort_values().values
            n_s = len(pvals_s)
            exp_s = -np.log10(np.arange(1, n_s+1) / (n_s+1))
            obs_s = -np.log10(pvals_s)
            qq_s = pd.DataFrame({"Expected": sorted(exp_s),
                                 "Observed": sorted(obs_s, reverse=True)})
            fig_q = px.scatter(qq_s, x="Expected", y="Observed",
                               title="QQ Plot")
            mx_s = max(qq_s["Expected"].max(), qq_s["Observed"].max())
            fig_q.add_shape(type="line", x0=0, y0=0, x1=mx_s, y1=mx_s,
                            line=dict(color="red", dash="dash"))
            fig_q = add_common_layout_options(fig_q, height=550)
            st.plotly_chart(fig_q, use_container_width=True)

            download_plotly_html(fig_m, "snp_manhattan.html", key="dl_snp_m")
            download_plotly_html(fig_q, "snp_qq.html", key="dl_snp_q")

        else:
            # Genotype matrix analysis
            num_s = numeric_columns(df_s)
            sample_col_s = st.selectbox("Sample ID column",
                                        ["None"] + df_s.columns.tolist(),
                                        key="snp_sid")
            group_col_s = st.selectbox("Group column",
                                       ["None"] + df_s.columns.tolist(),
                                       key="snp_grp")
            sample_col_s = None if sample_col_s == "None" else sample_col_s
            group_col_s = None if group_col_s == "None" else group_col_s

            snp_cols = st.multiselect("SNP marker columns", num_s,
                                      default=num_s[:min(200, len(num_s))],
                                      key="snp_markers")
            if len(snp_cols) < 2:
                st.warning("Select ≥ 2 SNP columns.")
                st.stop()

            mat_s = build_numeric_matrix(df_s, sample_col_s, snp_cols, "Mean")

            # Quality summaries
            miss_r = df_s[snp_cols].isna().mean()
            allele_p = mat_s.mean(axis=0) / 2
            maf = allele_p.apply(lambda p: min(p, 1-p))
            stats_s = pd.DataFrame({"Marker": snp_cols,
                                    "MissingRate": miss_r.values,
                                    "MAF": maf.values})

            st.subheader("SNP Quality")
            qc1, qc2 = st.columns(2)
            with qc1:
                fig_mr = px.histogram(stats_s, x="MissingRate", nbins=40,
                                      title="Marker missing rate")
                st.plotly_chart(fig_mr, use_container_width=True)
            with qc2:
                fig_maf = px.histogram(stats_s, x="MAF", nbins=40,
                                       title="Minor allele frequency")
                st.plotly_chart(fig_maf, use_container_width=True)

            # PCA
            st.subheader("SNP PCA")
            X_snp = StandardScaler().fit_transform(mat_s.values)
            pca_s = PCA(n_components=2)
            scores_s = pca_s.fit_transform(X_snp)
            pca_df_s = pd.DataFrame(scores_s, columns=["PC1", "PC2"])
            pca_df_s["Sample"] = mat_s.index.astype(str)

            color_s = None
            if group_col_s and sample_col_s:
                meta_s = (df_s[[sample_col_s, group_col_s]]
                          .drop_duplicates().set_index(sample_col_s))
                pca_df_s[group_col_s] = (pca_df_s["Sample"]
                                         .map(meta_s[group_col_s].astype(str)))
                color_s = group_col_s

            fig_pca_s = px.scatter(pca_df_s, x="PC1", y="PC2",
                                   color=color_s, hover_data=["Sample"],
                                   title="SNP-based PCA")
            fig_pca_s = add_common_layout_options(fig_pca_s, height=600)
            st.plotly_chart(fig_pca_s, use_container_width=True)

            download_plotly_html(fig_pca_s, "snp_pca.html", key="dl_snp_pca")
            download_dataframe(stats_s, "snp_stats.csv", key="dl_snp_stats")

# ═══════════════════════════════════════════
# TAB 3 — Phylogenetic Analysis
# ═══════════════════════════════════════════
with tab_phylo:
    st.subheader("🌳 Phylogenetic / Dendrogram Analysis")
    st.write(
        "Upload a numeric matrix (samples × features). "
        "A distance-based dendrogram will be constructed."
    )

    df_p = load_data_widget("phylo", "Upload data for phylogenetic tree")
    if df_p is not None:
        show_dataframe_overview(df_p)
        num_p = numeric_columns(df_p)
        if len(num_p) < 2:
            st.error("Need ≥ 2 numeric columns.")
            st.stop()

        sample_col_p = st.selectbox("Sample / taxon ID column",
                                     ["None"] + df_p.columns.tolist(),
                                     key="phylo_sid")
        sample_col_p = None if sample_col_p == "None" else sample_col_p

        feat_p = st.multiselect("Feature columns", num_p,
                                default=num_p[:min(50, len(num_p))],
                                key="phylo_feat")
        if len(feat_p) < 2:
            st.warning("Select ≥ 2 features.")
            st.stop()

        mat_p = build_numeric_matrix(df_p, sample_col_p, feat_p, "Mean")

        dist_metric = st.selectbox("Distance metric",
                                   ["euclidean", "cosine", "correlation",
                                    "cityblock", "chebyshev"],
                                   key="phylo_dist")
        link_method = st.selectbox("Linkage method",
                                   ["ward", "complete", "average", "single"],
                                   key="phylo_link")

        scale_p = st.checkbox("Standardize features", True, key="phylo_scale")
        X_p = mat_p.values
        if scale_p:
            X_p = StandardScaler().fit_transform(X_p)

        # Only ward works with euclidean
        if link_method == "ward":
            dist_metric = "euclidean"

        dist_mat = pdist(X_p, metric=dist_metric)
        Z = linkage(dist_mat, method=link_method)
        labels_p = mat_p.index.astype(str).tolist()

        # Build Plotly dendrogram
        from scipy.cluster.hierarchy import dendrogram as sci_dendro
        import plotly.figure_factory as ff

        if len(labels_p) <= 300:
            fig_dendro = ff.create_dendrogram(
                X_p, labels=labels_p, orientation="left",
                linkagefun=lambda x: Z,
                distfun=lambda x: dist_mat,
            )
            fig_dendro.update_layout(
                title="Phylogenetic Dendrogram",
                template="plotly_white",
                height=max(600, len(labels_p) * 18),
                xaxis_title="Distance",
            )
            st.plotly_chart(fig_dendro, use_container_width=True)
            download_plotly_html(fig_dendro, "phylogenetic_tree.html",
                                key="dl_phylo_tree")
        else:
            st.warning("Too many samples for interactive dendrogram "
                       "(>300). Showing distance heatmap instead.")

        # Distance heatmap
        from scipy.spatial.distance import squareform
        dist_sq = squareform(dist_mat)
        dist_df = pd.DataFrame(dist_sq, index=labels_p, columns=labels_p)

        st.subheader("Distance Matrix Heatmap")
        fig_dist = px.imshow(dist_df, color_continuous_scale="Viridis",
                             aspect="auto",
                             title="Pairwise Distance Heatmap")
        fig_dist.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)

        download_dataframe(dist_df.reset_index(), "distance_matrix.csv",
                           index=True, key="dl_phylo_dist")

        # Newick export
        tree_root = to_tree(Z)

        def _to_newick(node, labels):
            if node.is_leaf():
                return labels[node.id]
            left = _to_newick(node.get_left(), labels)
            right = _to_newick(node.get_right(), labels)
            return f"({left}:{node.dist/2:.4f},{right}:{node.dist/2:.4f})"

        newick = _to_newick(tree_root, labels_p) + ";"
        st.subheader("Newick Format")
        st.code(newick, language="text")
        st.download_button("Download Newick (.nwk)", newick,
                           "tree.nwk", "text/plain", key="dl_newick")