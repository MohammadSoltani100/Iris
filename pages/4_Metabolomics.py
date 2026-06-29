"""
Metabolomics analysis page.
Supports long and wide formats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    download_plotly_html, download_dataframe,
    add_common_layout_options,
)

st.title("⚗️ Metabolomics Analysis")
st.markdown("---")
st.write("Upload metabolomics data in **long** or **wide** format.")

df = load_data_widget("metabolomics", "Upload metabolomics data")
if df is None:
    st.info("👈 Please upload a file to begin.")
    st.stop()

show_dataframe_overview(df)

data_format = st.radio("Data format",
                       ["Long format", "Wide format"], horizontal=True)

# ════════════════════════════════════════════
# LONG FORMAT
# ════════════════════════════════════════════
if data_format == "Long format":
    c1, c2, c3 = st.columns(3)
    with c1:
        metab_col = st.selectbox("Metabolite column",
                                 df.columns.tolist(), key="m_met")
    with c2:
        val_col = st.selectbox("Concentration column",
                               numeric_columns(df), key="m_val")
    with c3:
        grp_col = st.selectbox("Group / treatment column",
                               df.columns.tolist(), key="m_grp")

    metab_list = sorted(df[metab_col].dropna().astype(str).unique())
    sel_met = st.multiselect(
        "Select metabolites", metab_list,
        default=metab_list[:min(8, len(metab_list))], key="m_sel")
    plot_df = df[df[metab_col].astype(str).isin(sel_met)].copy()

    plot_type = st.selectbox(
        "Plot type",
        ["Box plot", "Violin plot", "Mean bar plot",
         "Heatmap", "Radar plot", "PCA"],
        key="m_pt")

    if plot_type == "Box plot":
        fig = px.box(plot_df, x=metab_col, y=val_col,
                     color=grp_col, points="outliers")
    elif plot_type == "Violin plot":
        fig = px.violin(plot_df, x=metab_col, y=val_col,
                        color=grp_col, box=True, points="all")
    elif plot_type == "Mean bar plot":
        summ = plot_df.groupby([metab_col, grp_col],
                               as_index=False)[val_col].mean()
        fig = px.bar(summ, x=metab_col, y=val_col,
                     color=grp_col, barmode="group")
    elif plot_type == "Heatmap":
        pv = plot_df.pivot_table(index=metab_col, columns=grp_col,
                                  values=val_col, aggfunc="mean")
        fig = px.imshow(pv, text_auto=".2f",
                        color_continuous_scale="Viridis", aspect="auto")
    elif plot_type == "Radar plot":
        group_choice = st.selectbox(
            "Group", sorted(plot_df[grp_col].dropna().astype(str).unique()),
            key="m_rad")
        rd = (plot_df[plot_df[grp_col].astype(str) == group_choice]
              .groupby(metab_col)[val_col].mean())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=rd.values, theta=rd.index,
                                      fill="toself", name=group_choice))
    else:  # PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        pv = plot_df.pivot_table(index=grp_col, columns=metab_col,
                                  values=val_col, aggfunc="mean").fillna(0)
        if pv.shape[0] >= 2 and pv.shape[1] >= 2:
            Xm = StandardScaler().fit_transform(pv.values)
            pc = PCA(n_components=2).fit_transform(Xm)
            pcdf = pd.DataFrame(pc, columns=["PC1", "PC2"])
            pcdf["Group"] = pv.index.astype(str)
            fig = px.scatter(pcdf, x="PC1", y="PC2", text="Group",
                             color="Group",
                             title="Metabolomics PCA (by group means)")
            fig.update_traces(textposition="top center",
                              marker=dict(size=10))
        else:
            st.warning("Need ≥ 2 groups and ≥ 2 metabolites for PCA.")
            st.stop()

    fig = add_common_layout_options(fig, height=650)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "metabolomics_plot.html", key="dl_m_p")
    download_dataframe(plot_df, "metabolomics_filtered.csv",
                       key="dl_m_d")

# ════════════════════════════════════════════
# WIDE FORMAT
# ════════════════════════════════════════════
else:
    num_w = numeric_columns(df)
    s_col = st.selectbox("Sample ID column",
                         ["None"] + df.columns.tolist(), key="mw_sid")
    g_col = st.selectbox("Group column",
                         ["None"] + df.columns.tolist(), key="mw_grp")
    s_col = None if s_col == "None" else s_col
    g_col = None if g_col == "None" else g_col

    m_cols = st.multiselect("Metabolite columns", num_w,
                            default=num_w[:min(10, len(num_w))],
                            key="mw_mets")
    if len(m_cols) < 2:
        st.warning("Select ≥ 2 metabolite columns.")
        st.stop()

    melted = df.melt(
        id_vars=[c for c in [s_col, g_col] if c],
        value_vars=m_cols,
        var_name="Metabolite", value_name="Concentration")

    plot_w = st.selectbox("Plot type",
                          ["Box plot", "Violin plot",
                           "Heatmap (means)", "PCA"], key="mw_pt")

    if plot_w == "Box plot":
        fig = px.box(melted, x="Metabolite", y="Concentration",
                     color=g_col if g_col else None, points="outliers")
    elif plot_w == "Violin plot":
        fig = px.violin(melted, x="Metabolite", y="Concentration",
                        color=g_col if g_col else None,
                        box=True, points="all")
    elif plot_w == "Heatmap (means)":
        if g_col is None:
            st.warning("Heatmap requires a group column.")
            st.stop()
        pv = melted.pivot_table(index="Metabolite", columns=g_col,
                                 values="Concentration", aggfunc="mean")
        fig = px.imshow(pv, text_auto=".2f",
                        color_continuous_scale="Viridis", aspect="auto")
    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        Xw = df[m_cols].fillna(df[m_cols].mean()).values
        Xw = StandardScaler().fit_transform(Xw)
        pc = PCA(n_components=2).fit_transform(Xw)
        pcdf = pd.DataFrame(pc, columns=["PC1", "PC2"])
        if s_col:
            pcdf["Sample"] = df[s_col].astype(str).values
        if g_col:
            pcdf[g_col] = df[g_col].astype(str).values
        fig = px.scatter(pcdf, x="PC1", y="PC2",
                         color=g_col if g_col else None,
                         text="Sample" if s_col else None,
                         title="Metabolomics Sample PCA")
        fig.update_traces(marker=dict(size=10),
                          textposition="top center")

    fig = add_common_layout_options(fig, height=650)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly_html(fig, "metabolomics_wide_plot.html",
                         key="dl_mw_p")
    download_dataframe(melted, "metabolomics_long_converted.csv",
                       key="dl_mw_d")