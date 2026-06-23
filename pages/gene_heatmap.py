import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Gene Expression Analysis", page_icon="🧬", layout="wide")

st.title("🧬 تحلیل بیان ژن")

# تولید داده نمونه
@st.cache_data
def generate_expression_data():
    np.random.seed(42)
    genes = [f'AT{i}G{str(j).zfill(5)}' for i, j in 
             zip(np.random.randint(1, 6, 50), np.random.randint(10000, 99999, 50))]
    conditions = ['Control', 'Drought_1h', 'Drought_6h', 'Drought_12h', 'Drought_24h', 'Recovery']
    
    data = np.random.randn(50, 6) * 2 + np.random.randn(50, 1)  # بیان ژن
    df = pd.DataFrame(data, index=genes, columns=conditions)
    return df

df = generate_expression_data()

# سایدبار تنظیمات
st.sidebar.header("⚙️ تنظیمات Heatmap")
colormap = st.sidebar.selectbox("Color Map:", ["RdYlGn", "viridis", "coolwarm", "YlOrRd"])
cluster_rows = st.sidebar.checkbox("Cluster Rows", True)
cluster_cols = st.sidebar.checkbox("Cluster Columns", False)
n_genes = st.sidebar.slider("تعداد ژن‌ها:", 10, 50, 30)

# رسم Heatmap با Seaborn
st.subheader("📊 Heatmap بیان ژن (Seaborn)")

fig, ax = plt.subplots(figsize=(12, 10))
sns.clustermap(
    df.head(n_genes),
    cmap=colormap,
    row_cluster=cluster_rows,
    col_cluster=cluster_cols,
    standard_scale=1,
    figsize=(12, 10)
)
st.pyplot(plt.gcf())
plt.close()

# رسم Heatmap تعاملی با Plotly
st.subheader("📊 Heatmap تعاملی (Plotly)")

fig = px.imshow(
    df.head(n_genes),
    labels=dict(x="Condition", y="Gene", color="Expression"),
    color_continuous_scale=colormap,
    aspect="auto"
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)