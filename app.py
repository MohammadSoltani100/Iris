# app.py - اپلیکیشن رسم نمودار
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# تنظیمات صفحه
st.set_page_config(
    page_title="رسم نمودار",
    page_icon="📊",
    layout="wide"
)

st.title("📊 اپلیکیشن رسم نمودار")
st.write("فایل CSV یا Excel خود را آپلود کنید و نمودار رسم کنید!")

# ========== بخش آپلود فایل ==========
uploaded_file = st.file_uploader(
    "فایل داده را انتخاب کنید:",
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:
    # خواندن فایل
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # نمایش داده‌ها
    st.subheader("📋 پیش‌نمایش داده‌ها")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write(f"**تعداد سطرها:** {len(df)} | **تعداد ستون‌ها:** {len(df.columns)}")
    
    # ========== انتخاب نوع نمودار ==========
    st.subheader("📈 انتخاب نوع نمودار")
    
    chart_type = st.selectbox(
        "نوع نمودار:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"]
    )
    
    # ستون‌های عددی
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.error("فایل شما باید حداقل یک ستون عددی داشته باشد!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("محور X:", df.columns.tolist())
        with col2:
            y_col = st.selectbox("محور Y:", numeric_cols)
        
        # رسم نمودار
        st.subheader("📊 نمودار")
        
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Heatmap":
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 لطفاً یک فایل CSV یا Excel آپلود کنید")
    
    # نمایش داده نمونه
    st.subheader("📌 یا از داده نمونه استفاده کنید:")
    if st.button("بارگذاری داده نمونه"):
        # داده Iris
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]
        st.session_state['sample_data'] = df
        st.rerun()