import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.subheader("📊 PCA - کاهش ابعاد")

@st.cache_data
def generate_pca_data():
    np.random.seed(42)
    n_samples = 30
    n_genes = 100
    
    # سه گروه تیمار
    control = np.random.randn(10, n_genes)
    drought = np.random.randn(10, n_genes) + 2
    salt = np.random.randn(10, n_genes) - 1
    
    data = np.vstack([control, drought, salt])
    
    groups = ['Control'] * 10 + ['Drought'] * 10 + ['Salt'] * 10
    samples = [f'Sample_{i}' for i in range(n_samples)]
    
    return data, groups, samples

data, groups, samples = generate_pca_data()

# PCA
n_components = st.slider("تعداد مولفه‌ها:", 2, 5, 2)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(data_scaled)

# DataFrame برای رسم
pca_df = pd.DataFrame(
    pca_result[:, :2],
    columns=['PC1', 'PC2']
)
pca_df['Treatment'] = groups
pca_df['Sample'] = samples

# رسم نمودار
fig = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    color='Treatment',
    text='Sample',
    title=f'PCA Plot (Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})',
    color_discrete_map={'Control': '#22c55e', 'Drought': '#ef4444', 'Salt': '#3b82f6'}
)

fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Scree Plot
st.subheader("📉 Scree Plot")
variance_explained = pca.explained_variance_ratio_

fig2 = px.bar(
    x=[f'PC{i+1}' for i in range(n_components)],
    y=variance_explained,
    title='Variance Explained by Components',
    labels={'x': 'Principal Component', 'y': 'Variance Explained'}
)
st.plotly_chart(fig2, use_container_width=True)