"""
Phenomics analysis page.
Trait correlations, distributions, scatter matrix.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("🌿 Phenomics Analysis")
st.markdown("---")
st.write("Upload phenotype data. Rows: genotypes / plots / samples.")

df = load_data_widget("phenomics", "Upload phenomics data")
if df is None:
    st.info("👈 Please upload a file to begin.")
    st.stop()

show_dataframe_overview(df)

num_cols = numeric_columns(df)
if not num_cols:
    st.error("No numeric phenotype traits found.")
    st.stop()

group_col = st.selectbox("Group column (for coloring / comparison)",
                          ["None"] + df.columns.tolist(), key="ph_grp")
group_col = None if group_col == "None" else group_col

# ── Correlation heatmap ──
st.subheader("Trait correlation heatmap")
cc = st.multiselect("Traits for correlation", num_cols,
                    default=num_cols[:min(12, len(num_cols))],
                    key="ph_cc")
cm = st.selectbox("Correlation method",
                  ["pearson", "spearman", "kendall"], key="ph_cm")

if len(cc) >= 2:
    corr = df[cc].corr(method=cm)
    fig_c = px.imshow(corr, text_auto=".2f",
                      color_continuous_scale="RdBu_r", aspect="auto",
                      title=f"Trait Correlation ({cm})")
    fig_c.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_c, use_container_width=True)
    download_plotly_html(fig_c, "phenomics_corr.html", key="dl_ph_c")

# ── Visualisation ──
st.subheader("Trait Visualization")
pt = st.selectbox("Plot type",
                  ["Histogram + Box", "Box by group",
                   "Scatter", "Scatter matrix"], key="ph_pt")

if pt == "Histogram + Box":
    tr = st.selectbox("Trait", num_cols, key="ph_tr1")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Histogram", "Box plot"])
    fig.add_trace(go.Histogram(x=df[tr]), row=1, col=1)
    fig.add_trace(go.Box(y=df[tr]), row=1, col=2)
    fig.update_layout(template="plotly_white", height=500,
                      title=f"Distribution of {tr}")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "pheno_dist.html", key="dl_ph_d")

elif pt == "Box by group":
    if group_col is None:
        st.warning("Select a group column first.")
    else:
        tr = st.selectbox("Trait", num_cols, key="ph_tr2")
        fig = px.box(df, x=group_col, y=tr, color=group_col, points="all")
        fig.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_html(fig, "pheno_boxgrp.html", key="dl_ph_bg")

elif pt == "Scatter":
    c1, c2 = st.columns(2)
    with c1:
        xc = st.selectbox("X trait", num_cols, key="ph_xc")
    with c2:
        yc = st.selectbox("Y trait", num_cols,
                          index=min(1, len(num_cols)-1), key="ph_yc")
    fig = px.scatter(df, x=xc, y=yc, color=group_col,
                     hover_data=df.columns,
                     trendline="ols" if len(df) >= 5 else None)
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "pheno_scatter.html", key="dl_ph_sc")

else:
    tt = st.multiselect("Traits", num_cols,
                        default=num_cols[:min(4, len(num_cols))],
                        key="ph_tt")
    if len(tt) >= 2:
        fig = px.scatter_matrix(df, dimensions=tt, color=group_col)
        fig.update_layout(template="plotly_white", height=800)
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_html(fig, "pheno_matrix.html", key="dl_ph_m")

# ── Descriptive ──
st.subheader("Descriptive statistics")
desc = (df[num_cols].describe().T.reset_index()
        .rename(columns={"index": "Trait"}))
st.dataframe(desc, use_container_width=True)
download_dataframe(desc, "pheno_descriptive.csv", key="dl_ph_desc")