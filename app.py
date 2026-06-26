import streamlit as st

# ─────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Omics Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Define all pages with correct display names
# ─────────────────────────────────────────────
pages = {
    "🏠 Home": [
        st.Page("pages/Home.py", title="Home", icon="🏠"),
    ],
    "🧬 Omics Analysis": [
        st.Page("pages/Genomics.py",          title="Genomics",          icon="🧬"),
        st.Page("pages/Metabolomics.py",      title="Metabolomics",      icon="⚗️"),
        st.Page("pages/Phenomics.py",         title="Phenomics",         icon="🌿"),
        st.Page("pages/Gene_Heatmap.py",      title="Gene Heatmap",      icon="🔥"),
        st.Page("pages/Metabolic_Pathway.py", title="Metabolic Pathway", icon="🔄"),
        st.Page("pages/SNP_Analysis.py",      title="SNP Analysis",      icon="🧪"),
    ],
    "📊 Dimensionality Reduction & Clustering": [
        st.Page("pages/PCA_Analysis.py",      title="PCA Analysis",      icon="📉"),
        st.Page("pages/UMAP_Analysis.py",     title="UMAP Analysis",     icon="🗺️"),
        st.Page("pages/Cluster_Analysis.py",  title="Cluster Analysis",  icon="🔵"),
        st.Page("pages/Volcano_Plot.py",      title="Volcano Plot",      icon="🌋"),
    ],
    "🤖 Machine Learning": [
        st.Page("pages/Regression.py",        title="Regression",        icon="📈"),
        st.Page("pages/Classification.py",    title="Classification",    icon="🏷️"),
        st.Page("pages/Feature_Selection.py", title="Feature Selection", icon="🎯"),
    ],
}

# ─────────────────────────────────────────────
# Run navigation
# ─────────────────────────────────────────────
pg = st.navigation(pages)
pg.run()