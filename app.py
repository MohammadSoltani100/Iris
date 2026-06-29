import streamlit as st

st.set_page_config(
    page_title="Multi-Omics Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold;
        color: #1E88E5; text-align: center; margin-bottom: .5rem;
    }
    .sub-header {
        font-size: 1.2rem; color: #666;
        text-align: center; margin-bottom: 2rem;
    }
    .cat-box {
        background: #f8f9fa; border-left: 4px solid #1E88E5;
        padding: 1rem 1.5rem; border-radius: 6px; margin-bottom: 1rem;
    }
    /* Hide the default "app" entry that duplicates Home */
    [data-testid="stSidebarNav"] li:first-child { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 Multi-Omics Analysis Platform</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Comprehensive genomics, transcriptomics, proteomics, '
    'metabolomics, phenomics &amp; machine-learning analysis</p>',
    unsafe_allow_html=True)
st.markdown("---")

st.header("👋 Welcome")
st.markdown("""
Use the **sidebar** to navigate between analysis modules.  
Each module accepts **CSV or Excel** files.  
For Excel files you can **choose the sheet** to load.
""")

st.header("📂 Available Modules")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="cat-box">', unsafe_allow_html=True)
    st.markdown("### 🧬 Omics")
    st.markdown("""
- **Genomics** → GWAS · SNP · Phylogenetic
- **Transcriptomics** → DE · Heatmap · Volcano
- **Proteomics** → Protein quantification
- **Metabolomics** → Metabolite profiling
- **Phenomics** → Trait analysis
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="cat-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Multivariate")
    st.markdown("""
- **PCA Analysis**
- **Cluster Analysis**
- **UMAP Analysis**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="cat-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 Machine Learning")
    st.markdown("""
- **Regression** — Linear · Poly · RF · XGB
- **Classification** — Logistic · SVM · RF · XGB
- **Feature Selection** — Forward · Backward · Lasso · Ridge
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.header("🚀 Quick Start")
st.markdown("""
1. Pick a module from the sidebar  
2. Upload CSV / Excel (choose sheet if Excel)  
3. Configure parameters  
4. Click **Run**  
5. Download results
""")

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;padding:1rem;">'
    '🧬 Multi-Omics Analysis Platform | Built with Streamlit</div>',
    unsafe_allow_html=True)