import streamlit as st

st.set_page_config(
    page_title="🧬 Multi-Omics Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
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
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-badge {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Page ---
st.markdown('<p class="main-header">🧬 Multi-Omics Analysis Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A comprehensive platform for genomics, metabolomics, phenomics, and machine learning analysis</p>', unsafe_allow_html=True)

st.markdown("---")

# --- Overview ---
st.header("🏠 Welcome")
st.markdown("""
This platform provides a wide range of bioinformatics and machine learning analysis tools.  
Use the **sidebar** to navigate between different analysis modules.
""")

# --- Analysis Categories ---
st.header("📂 Available Analysis Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧬 Omics Analysis")
    st.markdown("""
    - **Genomics** — Genomic data analysis
    - **Metabolomics** — Metabolomics analysis
    - **Phenomics** — Phenomics analysis
    - **Gene Heatmap** — Gene expression heatmap
    - **Metabolic Pathway** — Pathway visualization
    - **SNP Analysis** — SNP data analysis
    """)

with col2:
    st.markdown("### 📊 Dimensionality Reduction & Clustering")
    st.markdown("""
    - **PCA Analysis** — Principal Component Analysis
    - **UMAP Analysis** — Uniform Manifold Approximation
    - **Cluster Analysis** — Clustering analysis
    - **Volcano Plot** — Differential expression visualization
    """)

with col3:
    st.markdown("### 🤖 Machine Learning")
    st.markdown("""
    - **Regression** — Linear, Polynomial, RF, XGBoost
    - **Classification** — Logistic, SVM, RF, XGBoost
    - **Feature Selection** — Forward, Backward, Lasso, Ridge
    """)

st.markdown("---")

# --- Quick Start ---
st.header("🚀 Quick Start Guide")
st.markdown("""
1. **Select an analysis** from the sidebar navigation
2. **Upload your CSV data** file
3. **Configure parameters** in the sidebar
4. **Click Run** to execute the analysis
5. **Download results** and visualizations
""")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>🧬 Multi-Omics Analysis Platform | Built with Streamlit</p>
    <p>For research and educational purposes</p>
</div>
""", unsafe_allow_html=True)