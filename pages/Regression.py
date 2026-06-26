import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page title (set_page_config is in app.py)
# ─────────────────────────────────────────────
st.title("📈 Regression Analysis")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar settings
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Regression Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # ── Load data ─────────────────────────────
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Show basic info
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Total Rows",    df.shape[0])
    ci2.metric("Total Columns", df.shape[1])
    ci3.metric("Missing Values", df.isnull().sum().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("❌ Dataset must have at least 2 numeric columns.")
        st.stop()

    # ── Target variable ───────────────────────
    target_col = st.sidebar.selectbox(
        "Select Target Variable (y)", options=numeric_cols
    )

    # ── Feature variables ─────────────────────
    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Select Feature Variables (X)",
        options=available_features,
        default=available_features,
    )

    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature variable.")
        st.stop()

    # ── Regression method ─────────────────────
    reg_type = st.sidebar.selectbox(
        "Select Regression Method",
        [
            "Linear Regression",
            "Multi-Linear Regression",
            "Polynomial Regression",
            "Random Forest Regression",
            "XGBoost Regression",
        ],
    )

    # ── Method-specific hyper-parameters ──────

    poly_degree = 2
    if reg_type == "Polynomial Regression":
        poly_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)

    rf_n_est, rf_max_depth = 100, None
    if reg_type == "Random Forest Regression":
        rf_n_est     = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
        rf_max_depth = st.sidebar.selectbox(
            "Max Depth", [None, 3, 5, 10, 15, 20]
        )

    xgb_lr, xgb_n_est, xgb_max_depth = 0.1, 100, 6
    if reg_type == "XGBoost Regression":
        xgb_lr        = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        xgb_n_est     = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        xgb_max_depth = st.sidebar.slider("Max Depth", 2, 15, 6, 1)

    # ── General settings ──────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Train / Test Split**")
    test_size    = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5) / 100
    random_state = int(st.sidebar.number_input("Random State", value=42, step=1))
    scale_data   = st.sidebar.checkbox("Standardize Features", value=True)

    st.sidebar.markdown("**Cross-Validation**")
    cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5, 1)

    # ─────────────────────────────────────────────
    # Run button
    # ─────────────────────────────────────────────
    run_btn = st.sidebar.button("🚀 Run Regression", use_container_width=True)

    if run_btn:

        # ── Resolve feature list for single/multi linear ──
        if reg_type == "Linear Regression" and len(feature_cols) > 1:
            st.warning(
                "⚠️ Linear Regression uses only the **first** selected feature. "
                "Choose Multi-Linear for multiple features."
            )
            feature_cols_used = [feature_cols[0]]
        elif reg_type == "Multi-Linear Regression" and len(feature_cols) < 2:
            st.warning(
                "⚠️ Multi-Linear Regression requires ≥ 2 features. "
                "Switched to Linear Regression."
            )
            feature_cols_used = feature_cols
            reg_type = "Linear Regression"
        else:
            feature_cols_used = feature_cols

        with st.spinner(f"Running {reg_type} …"):

            # ── Prepare raw data ───────────────────────
            data  = df[feature_cols_used + [target_col]].dropna()
            X_raw = data[feature_cols_used].values
            y     = data[target_col].values

            # ── Scale raw features ─────────────────────
            scaler = StandardScaler() if scale_data else None
            X_scaled = scaler.fit_transform(X_raw) if scaler else X_raw.copy()

            # ── Apply polynomial expansion (after scaling) ─
            poly = None
            if reg_type == "Polynomial Regression":
                poly      = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_scaled  = poly.fit_transform(X_scaled)

            # ── Train / Test split ─────────────────────
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=test_size,
                random_state=random_state,
            )

            # Show split sizes
            st.subheader("✂️ Train / Test Split")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Samples",    len(y))
            s2.metric("Training Samples", len(y_train),
                      f"{len(y_train)/len(y)*100:.1f}%")
            s3.metric("Test Samples",     len(y_test),
                      f"{len(y_test)/len(y)*100:.1f}%")

            # ── Build model ────────────────────────────
            if reg_type in ["Linear Regression",
                            "Multi-Linear Regression",
                            "Polynomial Regression"]:
                model = LinearRegression()

            elif reg_type == "Random Forest Regression":
                model = RandomForestRegressor(
                    n_estimators=rf_n_est,
                    max_depth=rf_max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            elif reg_type == "XGBoost Regression":
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=random_state,
                        n_jobs=-1, verbosity=0,
                    )
                except ImportError:
                    st.error("❌ XGBoost is not installed.")
                    st.stop()

            # ── Train ──────────────────────────────────
            model.fit(X_train, y_train)

            # ── Predict ────────────────────────────────
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            # ── Metrics ────────────────────────────────
            train_r2   = r2_score(y_train, y_train_pred)
            test_r2    = r2_score(y_test,  y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
            train_mae  = mean_absolute_error(y_train, y_train_pred)
            test_mae   = mean_absolute_error(y_test,  y_test_pred)

            try:
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
            except Exception:
                test_mape = np.nan

            # ── Cross-Validation ───────────────────────
            # Build a fresh pipeline for CV to avoid data-leakage
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            if reg_type == "Polynomial Regression":
                # Use Pipeline: scale → poly → linear
                cv_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly",   PolynomialFeatures(degree=poly_degree,
                                                  include_bias=False)),
                    ("reg",    LinearRegression()),
                ])
                cv_data_X = X_raw   # raw features, pipeline handles the rest
            else:
                # For non-poly models: scale inside pipeline
                if reg_type in ["Linear Regression", "Multi-Linear Regression"]:
                    base_reg = LinearRegression()
                elif reg_type == "Random Forest Regression":
                    base_reg = RandomForestRegressor(
                        n_estimators=rf_n_est,
                        max_depth=rf_max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                elif reg_type == "XGBoost Regression":
                    from xgboost import XGBRegressor
                    base_reg = XGBRegressor(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=random_state,
                        n_jobs=-1, verbosity=0,
                    )

                if scale_data:
                    cv_pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg",    base_reg),
                    ])
                else:
                    cv_pipe = base_reg

                cv_data_X = X_raw

            # Compute CV R² and RMSE
            cv_r2_scores  = cross_val_score(
                cv_pipe, cv_data_X, y, cv=kf, scoring="r2"
            )
            cv_neg_mse    = cross_val_score(
                cv_pipe, cv_data_X, y, cv=kf,
                scoring="neg_mean_squared_error"
            )
            cv_r2_mean,   cv_r2_std   = cv_r2_scores.mean(), cv_r2_scores.std()
            cv_rmse_scores = np.sqrt(-cv_neg_mse)
            cv_rmse_mean,  cv_rmse_std = cv_rmse_scores.mean(), cv_rmse_scores.std()

            st.success(f"✅ {reg_type} completed successfully!")

            # ─────────────────────────────────────────────
            # Section 1 — Performance Metrics
            # ─────────────────────────────────────────────
            st.subheader("📊 Performance Metrics")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train R²",   f"{train_r2:.4f}")
            m2.metric("Test R²",    f"{test_r2:.4f}")
            m3.metric("Train RMSE", f"{train_rmse:.4f}")
            m4.metric("Test RMSE",  f"{test_rmse:.4f}")

            metrics_table = pd.DataFrame({
                "Metric":   ["R²", "RMSE", "MAE", "MAPE (%)"],
                "Train":    [f"{train_r2:.4f}",
                             f"{train_rmse:.4f}",
                             f"{train_mae:.4f}", "—"],
                "Test":     [f"{test_r2:.4f}",
                             f"{test_rmse:.4f}",
                             f"{test_mae:.4f}",
                             f"{test_mape:.2f}" if not np.isnan(test_mape) else "N/A"],
                f"CV Mean ({cv_folds}-fold)": [
                    f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
                    f"{cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}",
                    "—", "—",
                ],
            })
            st.table(metrics_table)

            # ── CV fold-by-fold chart ──────────────────
            st.subheader(f"📉 Cross-Validation Results ({cv_folds} Folds)")

            cv_df = pd.DataFrame({
                "Fold":   [f"Fold {i+1}" for i in range(cv_folds)] * 2,
                "Score":  list(cv_r2_scores) + list(cv_rmse_scores),
                "Metric": ["R²"] * cv_folds + ["RMSE"] * cv_folds,
            })
            fig_cv = px.bar(
                cv_df, x="Fold", y="Score", color="Metric",
                barmode="group",
                title=f"CV Performance per Fold — {reg_type}",
                template="plotly_white",
                color_discrete_map={"R²": "#1E88E5", "RMSE": "#E53935"},
            )
            fig_cv.update_layout(height=420)
            st.plotly_chart(fig_cv, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 2 — Actual vs Predicted
            # ─────────────────────────────────────────────
            st.subheader("📈 Actual vs Predicted")

            fig_avp = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Training Set", "Test Set"),
            )

            # Training scatter
            fig_avp.add_trace(
                go.Scatter(
                    x=y_train, y=y_train_pred, mode="markers",
                    name="Train",
                    marker=dict(color="blue", opacity=0.6, size=8),
                ),
                row=1, col=1,
            )
            tr_min = min(y_train.min(), y_train_pred.min())
            tr_max = max(y_train.max(), y_train_pred.max())
            fig_avp.add_trace(
                go.Scatter(
                    x=[tr_min, tr_max], y=[tr_min, tr_max],
                    mode="lines", name="Perfect Fit",
                    line=dict(color="red", dash="dash"),
                ),
                row=1, col=1,
            )

            # Test scatter
            fig_avp.add_trace(
                go.Scatter(
                    x=y_test, y=y_test_pred, mode="markers",
                    name="Test",
                    marker=dict(color="green", opacity=0.6, size=8),
                ),
                row=1, col=2,
            )
            te_min = min(y_test.min(), y_test_pred.min())
            te_max = max(y_test.max(), y_test_pred.max())
            fig_avp.add_trace(
                go.Scatter(
                    x=[te_min, te_max], y=[te_min, te_max],
                    mode="lines", name="Perfect Fit",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=1, col=2,
            )

            fig_avp.update_layout(
                height=500, template="plotly_white",
                title_text=f"{reg_type} — Actual vs Predicted",
                title_font_size=18,
            )
            fig_avp.update_xaxes(title_text="Actual",    row=1, col=1)
            fig_avp.update_yaxes(title_text="Predicted", row=1, col=1)
            fig_avp.update_xaxes(title_text="Actual",    row=1, col=2)
            fig_avp.update_yaxes(title_text="Predicted", row=1, col=2)
            st.plotly_chart(fig_avp, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 3 — Residual Plot
            # ─────────────────────────────────────────────
            st.subheader("📉 Residual Plot (Test Set)")
            residuals = y_test - y_test_pred

            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_test_pred, y=residuals, mode="markers",
                marker=dict(color="purple", opacity=0.6, size=8),
                name="Residuals",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                title="Residuals vs Predicted",
                template="plotly_white", height=450,
            )
            st.plotly_chart(fig_res, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 4 — Residual Distribution
            # ─────────────────────────────────────────────
            st.subheader("📊 Residual Distribution (Test Set)")
            fig_hist = px.histogram(
                x=residuals, nbins=30,
                title="Distribution of Residuals",
                labels={"x": "Residual", "y": "Count"},
                template="plotly_white", opacity=0.7,
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 5 — Feature Importance / Coefficients
            # ─────────────────────────────────────────────
            if reg_type in ["Random Forest Regression", "XGBoost Regression"]:
                st.subheader("🏆 Feature Importance")
                imp_df = pd.DataFrame({
                    "Feature":    feature_cols_used,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=True)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h", title="Feature Importance",
                    template="plotly_white",
                    color="Importance",
                    color_continuous_scale="viridis",
                )
                fig_imp.update_layout(
                    height=max(400, len(feature_cols_used) * 30)
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            if reg_type in ["Linear Regression", "Multi-Linear Regression"]:
                st.subheader("📐 Model Coefficients")
                coef_vals = (
                    model.coef_
                    if model.coef_.ndim == 1
                    else model.coef_.flatten()
                )
                coef_df = pd.DataFrame({
                    "Feature":     feature_cols_used,
                    "Coefficient": coef_vals,
                })
                # Append intercept row
                intercept_row = pd.DataFrame({
                    "Feature":     ["Intercept"],
                    "Coefficient": [model.intercept_]
                })
                coef_df = pd.concat([coef_df, intercept_row], ignore_index=True)
                st.table(coef_df)

            # ─────────────────────────────────────────────
            # Section 6 — Download
            # ─────────────────────────────────────────────
            st.subheader("📥 Download Predictions")
            pred_df = pd.DataFrame({
                "Actual":    np.concatenate([y_train, y_test]),
                "Predicted": np.concatenate([y_train_pred, y_test_pred]),
                "Set":       ["Train"] * len(y_train) + ["Test"] * len(y_test),
            })
            st.download_button(
                label="Download Predictions CSV",
                data=pred_df.to_csv(index=False),
                file_name=f"regression_{reg_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    st.subheader("📌 Supported Methods")
    st.markdown("""
| Method | Description |
|--------|-------------|
| **Linear Regression** | Simple linear regression (1 feature) |
| **Multi-Linear Regression** | Multiple features, linear model |
| **Polynomial Regression** | Non-linear via polynomial features |
| **Random Forest Regression** | Ensemble tree-based regressor |
| **XGBoost Regression** | Gradient boosting regressor |

**Metrics:** R² · RMSE · MAE · MAPE · Cross-Validation (fold-by-fold chart)
""")