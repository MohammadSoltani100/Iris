import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.subheader("🌋 Volcano Plot - ژن‌های دیفرنشیال")

# تولید داده
@st.cache_data
def generate_volcano_data():
    np.random.seed(42)
    n_genes = 1000
    
    log2fc = np.random.normal(0, 1.5, n_genes)
    pvalue = np.random.uniform(0.0001, 1, n_genes)
    
    # افزودن ژن‌های معنادار
    significant_idx = np.random.choice(n_genes, 100, replace=False)
    log2fc[significant_idx] = np.random.choice([-1, 1], 100) * np.random.uniform(2, 5, 100)
    pvalue[significant_idx] = np.random.uniform(0.00001, 0.01, 100)
    
    df = pd.DataFrame({
        'Gene': [f'AT{i}G{str(j).zfill(5)}' for i, j in 
                 zip(np.random.randint(1, 6, n_genes), np.random.randint(10000, 99999, n_genes))],
        'log2FoldChange': log2fc,
        'pvalue': pvalue,
        '-log10(pvalue)': -np.log10(pvalue)
    })
    return df

df = generate_volcano_data()

# تنظیمات
col1, col2 = st.columns(2)
with col1:
    fc_threshold = st.slider("آستانه Fold Change:", 0.5, 3.0, 1.5, 0.1)
with col2:
    pval_threshold = st.slider("آستانه P-value:", 0.001, 0.1, 0.05, 0.001)

# تعیین وضعیت ژن‌ها
def classify_gene(row):
    if row['pvalue'] < pval_threshold:
        if row['log2FoldChange'] > fc_threshold:
            return 'Up-regulated'
        elif row['log2FoldChange'] < -fc_threshold:
            return 'Down-regulated'
    return 'Not Significant'

df['Status'] = df.apply(classify_gene, axis=1)

# رسم نمودار
colors = {'Up-regulated': '#ef4444', 'Down-regulated': '#3b82f6', 'Not Significant': '#6b7280'}

fig = px.scatter(
    df,
    x='log2FoldChange',
    y='-log10(pvalue)',
    color='Status',
    color_discrete_map=colors,
    hover_data=['Gene'],
    title='Volcano Plot - Differential Gene Expression'
)

# افزودن خطوط آستانه
fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="gray")
fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# نمایش آمار
col1, col2, col3 = st.columns(3)
col1.metric("Up-regulated", len(df[df['Status'] == 'Up-regulated']))
col2.metric("Down-regulated", len(df[df['Status'] == 'Down-regulated']))
col3.metric("Not Significant", len(df[df['Status'] == 'Not Significant']))