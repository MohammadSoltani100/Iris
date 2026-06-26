import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)

st.set_page_config(page_title="Regression Analysis", layout="wide")
st.title("📈 Regression Analysis")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("⚙️ Regression Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Dataset must have at least 2 numeric columns.")
        st.stop()

    # Target variable
    target_col = st.sidebar.selectbox("Select Target Variable (y)", options=numeric_cols)

    # Feature variables
    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Select Feature Variables (X)",
        options=available_features,
        default=available_features
    )

    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature variable.")
        st.stop()

    # Regression type
    reg_type = st.sidebar.selectbox(
        "Select Regression Method",
        [
            "Linear Regression",
            "Multi-Linear Regression",
            "Polynomial Regression",
            "Random Forest Regression",
            "XGBoost Regression"
        ]
    )

    # Additional parameters based on method
    poly_degree = 2
    if reg_type == "Polynomial Regression":
        poly_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)

    n_estimators_rf = 100
    max_depth_rf = None
    if reg_type == "Random Forest Regression":
        n_estimators_rf = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
        max_depth_rf = st.sidebar.selectbox("Max Depth", [None, 3, 5, 10, 15, 20])

    xgb_lr = 0.1
    xgb_n_est = 100
    xgb_max_depth = 6
    if reg_type == "XGBoost Regression":
        xgb_lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        xgb_n_est = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        xgb_max_depth = st.sidebar.slider("Max Depth", 2, 15, 6, 1)

    # Train-test split
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5) / 100
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    scale_features = st.sidebar.checkbox("Standardize Features", value=True)
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 2, 10, 5, 1)

    # --- Run Regression ---
    if st.sidebar.button("🚀 Run Regression", use_container_width=True):

        # Handle single feature for linear regression
        if reg_type == "Linear Regression" and len(feature_cols) > 1:
            st.warning("⚠️ Linear Regression uses only the first selected feature. Use Multi-Linear for multiple features.")
            feature_cols_used = [feature_cols[0]]
        elif reg_type == "Multi-Linear Regression" and len(feature_cols) < 2:
            st.warning("⚠️ Multi-Linear Regression requires at least 2 features. Switching to Linear Regression.")
            feature_cols_used = feature_cols
            reg_type = "Linear Regression"
        else:
            feature_cols_used = feature_cols

        with st.spinner(f"Running {reg_type}..."):
            # Prepare data
            data = df[feature_cols_used + [target_col]].dropna()
            X = data[feature_cols_used].values
            y = data[target_col].values

            # Polynomial features if needed
            if reg_type == "Polynomial Regression":
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X = poly.fit_transform(X)

            # Scale features
            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

            # Build model
            if reg_type in ["Linear Regression", "Multi-Linear Regression"]:
                model = LinearRegression()
            elif reg_type == "Polynomial Regression":
                model = LinearRegression()
            elif reg_type == "Random Forest Regression":
                model = RandomForestRegressor(
                    n_estimators=n_estimators_rf,
                    max_depth=max_depth_rf,
                    random_state=int(random_state),
                    n_jobs=-1
                )
            elif reg_type == "XGBoost Regression":
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=int(random_state),
                        n_jobs=-1,
                        verbosity=0
                    )
                except ImportError:
                    st.error("❌ XGBoost is not installed. Please add 'xgboost' to requirements.txt.")
                    st.stop()

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Safe MAPE calculation
            try:
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
            except Exception:
                test_mape = np.nan

            # Cross-validation
            if reg_type == "Polynomial Regression":
                # For polynomial, we need to redo the pipeline for CV
                cv_r2 = np.nan
                cv_rmse = np.nan
            else:
                # Rebuild X without polynomial for CV
                X_cv = data[feature_cols_used].values
                if scale_features:
                    X_cv = StandardScaler().fit_transform(X_cv)

                if reg_type in ["Linear Regression", "Multi-Linear Regression"]:
                    cv_model = LinearRegression()
                elif reg_type == "Random Forest Regression":
                    cv_model = RandomForestRegressor(
                        n_estimators=n_estimators_rf,
                        max_depth=max_depth_rf,
                        random_state=int(random_state),
                        n_jobs=-1
                    )
                elif reg_type == "XGBoost Regression":
                    from xgboost import XGBRegressor
                    cv_model = XGBRegressor(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=int(random_state),
                        n_jobs=-1,
                        verbosity=0
                    )

                cv_r2_scores = cross_val_score(cv_model, X_cv, y, cv=cv_folds, scoring='r2')
                cv_neg_mse = cross_val_score(cv_model, X_cv, y, cv=cv_folds, scoring='neg_mean_squared_error')
                cv_r2 = cv_r2_scores.mean()
                cv_rmse = np.sqrt(-cv_neg_mse.mean())

            st.success(f"✅ {reg_type} completed successfully!")

            # --- Display Metrics ---
            st.subheader("📊 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Train R²", f"{train_r2:.4f}")
            col2.metric("Test R²", f"{test_r2:.4f}")
            col3.metric("Train RMSE", f"{train_rmse:.4f}")
            col4.metric("Test RMSE", f"{test_rmse:.4f}")

            metrics_df = pd.DataFrame({
                "Metric": ["R²", "RMSE", "MAE", "MAPE (%)"],
                "Train": [f"{train_r2:.4f}", f"{train_rmse:.4f}", f"{train_mae:.4f}", "—"],
                "Test": [f"{test_r2:.4f}", f"{test_rmse:.4f}", f"{test_mae:.4f}", f"{test_mape:.2f}" if not np.isnan(test_mape) else "N/A"],
                "CV Mean": [
                    f"{cv_r2:.4f}" if not np.isnan(cv_r2) else "N/A",
                    f"{cv_rmse:.4f}" if not np.isnan(cv_rmse) else "N/A",
                    "—",
                    "—"
                ]
            })
            st.table(metrics_df)

            # --- Actual vs Predicted Plot ---
            st.subheader("📈 Actual vs Predicted")

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Set", "Test Set"))

            # Training set
            fig.add_trace(
                go.Scatter(
                    x=y_train, y=y_train_pred,
                    mode='markers',
                    name='Train',
                    marker=dict(color='blue', opacity=0.6, size=8),
                    showlegend=True
                ), row=1, col=1
            )

            # Perfect prediction line for training
            train_min = min(y_train.min(), y_train_pred.min())
            train_max = max(y_train.max(), y_train_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[train_min, train_max],
                    y=[train_min, train_max],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(color='red', dash='dash'),
                    showlegend=True
                ), row=1, col=1
            )

            # Test set
            fig.add_trace(
                go.Scatter(
                    x=y_test, y=y_test_pred,
                    mode='markers',
                    name='Test',
                    marker=dict(color='green', opacity=0.6, size=8),
                    showlegend=True
                ), row=1, col=2
            )

            # Perfect prediction line for test
            test_min = min(y_test.min(), y_test_pred.min())
            test_max = max(y_test.max(), y_test_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[test_min, test_max],
                    y=[test_min, test_max],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ), row=1, col=2
            )

            fig.update_layout(
                height=500, width=1000,
                template="plotly_white",
                title_text=f"{reg_type} - Actual vs Predicted",
                title_font_size=18
            )
            fig.update_xaxes(title_text="Actual", row=1, col=1)
            fig.update_yaxes(title_text="Predicted", row=1, col=1)
            fig.update_xaxes(title_text="Actual", row=1, col=2)
            fig.update_yaxes(title_text="Predicted", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)

            # --- Residual Plot ---
            st.subheader("📉 Residual Plot (Test Set)")
            residuals = y_test - y_test_pred

            fig_resid = go.Figure()
            fig_resid.add_trace(
                go.Scatter(
                    x=y_test_pred, y=residuals,
                    mode='markers',
                    marker=dict(color='purple', opacity=0.6, size=8),
                    name='Residuals'
                )
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            fig_resid.update_layout(
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                title="Residuals vs Predicted",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig_resid, use_container_width=True)

            # --- Residual Distribution ---
            st.subheader("📊 Residual Distribution")
            fig_hist = px.histogram(
                x=residuals,
                nbins=30,
                title="Distribution of Residuals",
                labels={"x": "Residual", "y": "Count"},
                template="plotly_white",
                opacity=0.7
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Feature Importance (for tree-based models) ---
            if reg_type in ["Random Forest Regression", "XGBoost Regression"]:
                st.subheader("🏆 Feature Importance")
                importances = model.feature_importances_

                if reg_type == "XGBoost Regression" or reg_type == "Random Forest Regression":
                    if reg_type == "Random Forest Regression":
                        feat_names = feature_cols_used
                    else:
                        feat_names = feature_cols_used

                imp_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=True)

                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title="Feature Importance",
                    template="plotly_white",
                    color="Importance",
                    color_continuous_scale="viridis"
                )
                fig_imp.update_layout(height=max(400, len(feat_names) * 30))
                st.plotly_chart(fig_imp, use_container_width=True)

            # --- Coefficients (for linear models) ---
            if reg_type in ["Linear Regression", "Multi-Linear Regression"]:
                st.subheader("📐 Model Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": feature_cols_used,
                    "Coefficient": model.coef_ if len(model.coef_.shape) == 1 else model.coef_.flatten()
                })
                coef_df.loc[len(coef_df)] = ["Intercept", model.intercept_]
                st.table(coef_df)

            # --- Download predictions ---
            st.subheader("📥 Download Predictions")
            pred_df = pd.DataFrame({
                "Actual": np.concatenate([y_train, y_test]),
                "Predicted": np.concatenate([y_train_pred, y_test_pred]),
                "Set": ["Train"] * len(y_train) + ["Test"] * len(y_test)
            })
            csv_pred = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv_pred,
                file_name=f"regression_predictions_{reg_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin Regression Analysis.")
    st.subheader("📌 Supported Methods")
    st.markdown("""
    | Method | Description |
    |--------|-------------|
    | **Linear Regression** | Simple linear regression with one feature |
    | **Multi-Linear Regression** | Multiple features, linear model |
    | **Polynomial Regression** | Non-linear using polynomial features |
    | **Random Forest Regression** | Ensemble tree-based method |
    | **XGBoost Regression** | Gradient boosting method |
    
    **Metrics displayed:** R², RMSE, MAE, MAPE, Cross-Validation scores
    """)