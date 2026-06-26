import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="UMAP Analysis", layout="wide")
st.title("🗺️ UMAP Analysis")
st.markdown("---")

# --- Sidebar settings ---
st.sidebar.header("⚙️ UMAP Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Select columns
    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Label column selection
    label_col = st.sidebar.selectbox(
        "Select Label Column (optional, for coloring)",
        options=["None"] + all_columns,
        index=0
    )

    # Feature columns selection
    feature_cols = st.sidebar.multiselect(
        "Select Feature Columns",
        options=numeric_cols,
        default=numeric_cols
    )

    if len(feature_cols) < 2:
        st.warning("⚠️ Please select at least 2 numeric feature columns.")
        st.stop()

    # UMAP hyperparameters
    n_neighbors = st.sidebar.slider("Number of Neighbors", 2, 100, 15, 1)
    min_dist = st.sidebar.slider("Minimum Distance", 0.0, 1.0, 0.1, 0.01)
    n_components = st.sidebar.radio("Number of Components", [2, 3], index=0)
    metric = st.sidebar.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan", "cosine", "chebyshev", "correlation"],
        index=0
    )
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    # --- Run UMAP ---
    if st.sidebar.button("🚀 Run UMAP", use_container_width=True):
        with st.spinner("Running UMAP dimensionality reduction..."):
            # Prepare data
            X = df[feature_cols].dropna()
            valid_idx = X.index

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit UMAP
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=int(random_state)
            )
            embedding = reducer.fit_transform(X_scaled)

            # Create result DataFrame
            if n_components == 2:
                umap_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
            else:
                umap_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2", "UMAP_3"])

            # Add label if selected
            color_col = None
            if label_col != "None":
                umap_df["Label"] = df.loc[valid_idx, label_col].values
                color_col = "Label"

            st.success("✅ UMAP completed successfully!")

            # --- Visualization ---
            st.subheader("📊 UMAP Visualization")

            if n_components == 2:
                fig = px.scatter(
                    umap_df,
                    x="UMAP_1",
                    y="UMAP_2",
                    color=color_col,
                    title="UMAP 2D Projection",
                    template="plotly_white",
                    width=900,
                    height=650,
                    opacity=0.7
                )
                fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
                fig.update_layout(
                    font=dict(size=14),
                    title_font_size=20,
                    xaxis_title="UMAP Component 1",
                    yaxis_title="UMAP Component 2"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter_3d(
                    umap_df,
                    x="UMAP_1",
                    y="UMAP_2",
                    z="UMAP_3",
                    color=color_col,
                    title="UMAP 3D Projection",
                    template="plotly_white",
                    width=900,
                    height=700,
                    opacity=0.7
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(
                    font=dict(size=14),
                    title_font_size=20
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Parameters summary ---
            st.subheader("📝 UMAP Parameters Used")
            params_df = pd.DataFrame({
                "Parameter": ["n_neighbors", "min_dist", "n_components", "metric", "random_state", "n_features"],
                "Value": [n_neighbors, min_dist, n_components, metric, int(random_state), len(feature_cols)]
            })
            st.table(params_df)

            # --- Download results ---
            st.subheader("📥 Download Results")
            csv = umap_df.to_csv(index=False)
            st.download_button(
                label="Download UMAP Results as CSV",
                data=csv,
                file_name="umap_results.csv",
                mime="text/csv",
                use_container_width=True
            )
else:
    st.info("👈 Please upload a CSV file from the sidebar to begin UMAP analysis.")

    # Example data section
    st.subheader("📌 Example")
    st.markdown("""
    **Expected CSV format:**
    
    | Sample | Feature1 | Feature2 | Feature3 | Group |
    |--------|----------|----------|----------|-------|
    | S1     | 5.1      | 3.5      | 1.4      | A     |
    | S2     | 4.9      | 3.0      | 1.4      | B     |
    | S3     | 7.0      | 3.2      | 4.7      | A     |
    
    - Select numeric columns as features
    - Optionally select a label column for coloring
    """)