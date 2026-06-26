import streamlit as st

# ─────────────────────────────────────────────
# Main home page content
# ─────────────────────────────────────────────

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-box {
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-header">🧬 Multi-Omics Analysis Platform</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">A comprehensive platform for genomics, metabolomics,'
    ' phenomics, and machine learning analysis</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ── Overview ──────────────────────────────────
st.header("👋 Welcome")
st.markdown("""
This platform provides a wide range of **bioinformatics** and **machine learning**
analysis tools.  
Use the **sidebar** to navigate between different analysis modules.
""")

# ── Analysis Categories ───────────────────────
st.header("📂 Available Analysis Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="category-box">', unsafe_allow_html=True)
    st.markdown("### 🧬 Omics Analysis")
    st.markdown("""
- **Genomics** — Genomic data analysis  
- **Metabolomics** — Metabolomics analysis  
- **Phenomics** — Phenomics analysis  
- **Gene Heatmap** — Expression heatmap  
- **Metabolic Pathway** — Pathway visualization  
- **SNP Analysis** — SNP data analysis  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="category-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Dimensionality Reduction")
    st.markdown("""
- **PCA Analysis** — Principal Component Analysis  
- **UMAP Analysis** — Uniform Manifold Approximation  
- **Cluster Analysis** — Clustering methods  
- **Volcano Plot** — Differential expression  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="category-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 Machine Learning")
    st.markdown("""
- **Regression** — Linear, Polynomial, RF, XGBoost  
- **Classification** — Logistic, SVM, RF, XGBoost  
- **Feature Selection** — Forward, Backward, Lasso, Ridge  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Quick Start ───────────────────────────────
st.header("🚀 Quick Start Guide")
st.markdown("""
1. **Select an analysis** from the sidebar navigation  
2. **Upload your CSV data** file  
3. **Configure parameters** in the sidebar  
4. **Click Run** to execute the analysis  
5. **Download results** and visualizations  
""")

# ── Data Format ───────────────────────────────
st.header("📋 Expected Data Format")
st.markdown("""
All analyses accept **CSV files** with the following general structure:

| Sample | Feature_1 | Feature_2 | Feature_3 | Group |
|--------|-----------|-----------|-----------|-------|
| S1     | 5.1       | 3.5       | 1.4       | A     |
| S2     | 4.9       | 3.0       | 1.4       | B     |
| S3     | 7.0       | 3.2       | 4.7       | A     |

- **Rows** = Samples / Observations  
- **Columns** = Features / Variables  
- **Group / Label column** = optional, for coloring and classification  
""")

# ── Footer ────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; padding:1rem;'>
    🧬 Multi-Omics Analysis Platform | Built with Streamlit
</div>
""", unsafe_allow_html=True)