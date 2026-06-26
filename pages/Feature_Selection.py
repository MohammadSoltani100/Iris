import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Feature Selection", layout="wide")
st.title("🎯 Feature Selection")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("⚙️ Feature Selection Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Target variable
    target_col = st.sidebar.selectbox("Select Target Variable (y)", options=all_columns)

    # Feature columns
    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Select Candidate Feature Variables (X)",
        options=available_features,
        default=available_features
    )

    if len(feature_cols) < 2:
        st.warning("⚠️ Please select at least 2 feature variables.")
        st.stop()

    # Task type
    task_type = st.sidebar.radio("Task Type", ["Regression", "Classification"])

    # Feature selection method
    fs_method = st.sidebar.selectbox(
        "Select Feature Selection Method",
        [
            "Forward Selection",
            "Backward Elimination",
            "Lasso (L1 Regularization)",
            "Ridge (L2 Regularization)"
        ]
    )

    # Parameters
    alpha_val = 1.0
    if fs_method in ["Lasso (L1 Regularization)", "Ridge (L2 Regularization)"]:
        alpha_val = st.sidebar.slider("Alpha (Regularization Strength)", 0.001, 10.0, 1.0, 0.001)

    cv_folds = st.sidebar.slider("Cross-Validation Folds", 2, 10, 5, 1)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    scale_features = st.sidebar.checkbox("Standardize Features", value=True)

    # --- Run Feature Selection ---
    if st.sidebar.button("🚀 Run Feature Selection", use_container_width=True):
        with st.spinner(f"Running {fs_method}..."):
            # Prepare data
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y_raw = data[target_col]

            # Encode target for classification
            le = None
            if task_type == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                scoring = 'accuracy'
                metric_name = "Accuracy"
            else:
                y = y_raw.values.astype(float)
                scoring = 'r2'
                metric_name = "R²"

            # Scale
            if scale_features:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=feature_cols,
                    index=X.index
                )
            else:
                X_scaled = X.copy()

            # --- FORWARD SELECTION ---
            if fs_method == "Forward Selection":
                selected_features = []
                remaining_features = list(feature_cols)
                history = []

                for step in range(len(feature_cols)):
                    best_score = -np.inf
                    best_feature = None

                    for feat in remaining_features:
                        current_features = selected_features + [feat]
                        X_subset = X_scaled[current_features].values

                        if task_type == "Classification":
                            model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                        else:
                            model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)

                        scores = cross_val_score(model, X_subset, y, cv=cv_folds, scoring=scoring)
                        mean_score = scores.mean()

                        if mean_score > best_score:
                            best_score = mean_score
                            best_feature = feat

                    if best_feature is not None:
                        selected_features.append(best_feature)
                        remaining_features.remove(best_feature)
                        history.append({
                            "Step": step + 1,
                            "Feature Added": best_feature,
                            f"CV {metric_name}": best_score,
                            "Selected Features": ", ".join(selected_features)
                        })

                history_df = pd.DataFrame(history)

                # Find optimal number of features
                best_step_idx = history_df[f"CV {metric_name}"].idxmax()
                optimal_features = selected_features[:best_step_idx + 1]
                best_metric_val = history_df[f"CV {metric_name}"].max()

            # --- BACKWARD ELIMINATION ---
            elif fs_method == "Backward Elimination":
                selected_features = list(feature_cols)
                history = []

                # Initial score with all features
                if task_type == "Classification":
                    model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)

                scores = cross_val_score(model, X_scaled[selected_features].values, y, cv=cv_folds, scoring=scoring)
                history.append({
                    "Step": 0,
                    "Feature Removed": "None (All Features)",
                    f"CV {metric_name}": scores.mean(),
                    "Remaining Features": ", ".join(selected_features)
                })

                for step in range(len(feature_cols) - 1):
                    best_score = -np.inf
                    worst_feature = None

                    for feat in selected_features:
                        current_features = [f for f in selected_features if f != feat]

                        if len(current_features) == 0:
                            continue

                        X_subset = X_scaled[current_features].values

                        if task_type == "Classification":
                            model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                        else:
                            model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)

                        scores = cross_val_score(model, X_subset, y, cv=cv_folds, scoring=scoring)
                        mean_score = scores.mean()

                        if mean_score > best_score:
                            best_score = mean_score
                            worst_feature = feat

                    if worst_feature is not None:
                        selected_features.remove(worst_feature)
                        history.append({
                            "Step": step + 1,
                            "Feature Removed": worst_feature,
                            f"CV {metric_name}": best_score,
                            "Remaining Features": ", ".join(selected_features)
                        })

                history_df = pd.DataFrame(history)
                best_step_idx = history_df[f"CV {metric_name}"].idxmax()

                # Reconstruct optimal features from history
                all_feats_copy = list(feature_cols)
                for i in range(1, best_step_idx + 1):
                    removed = history_df.iloc[i]["Feature Removed"]
                    if removed in all_feats_copy:
                        all_feats_copy.remove(removed)
                optimal_features = all_feats_copy
                best_metric_val = history_df[f"CV {metric_name}"].max()

            # --- LASSO ---
            elif fs_method == "Lasso (L1 Regularization)":
                if task_type == "Regression":
                    lasso = Lasso(alpha=alpha_val, random_state=int(random_state), max_iter=10000)
                    lasso.fit(X_scaled.values, y)
                    coefs = lasso.coef_
                else:
                    # For classification, use LogisticRegression with L1
                    log_l1 = LogisticRegression(
                        penalty='l1',
                        C=1.0 / alpha_val,
                        solver='saga',
                        max_iter=10000,
                        random_state=int(random_state)
                    )
                    log_l1.fit(X_scaled.values, y)
                    if log_l1.coef_.ndim > 1:
                        coefs = np.abs(log_l1.coef_).mean(axis=0)
                    else:
                        coefs = np.abs(log_l1.coef_[0])

                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": coefs,
                    "Abs_Coefficient": np.abs(coefs)
                }).sort_values("Abs_Coefficient", ascending=False)

                # Selected features: non-zero coefficients
                optimal_features = coef_df[coef_df["Abs_Coefficient"] > 1e-6]["Feature"].tolist()

                # Compute metric for selected features
                if len(optimal_features) > 0:
                    if task_type == "Classification":
                        eval_model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                    else:
                        eval_model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                    scores = cross_val_score(eval_model, X_scaled[optimal_features].values, y, cv=cv_folds, scoring=scoring)
                    best_metric_val = scores.mean()
                else:
                    best_metric_val = 0.0

                history_df = coef_df

            # --- RIDGE ---
            elif fs_method == "Ridge (L2 Regularization)":
                if task_type == "Regression":
                    ridge = Ridge(alpha=alpha_val)
                    ridge.fit(X_scaled.values, y)
                    coefs = ridge.coef_
                else:
                    log_l2 = LogisticRegression(
                        penalty='l2',
                        C=1.0 / alpha_val,
                        solver='lbfgs',
                        max_iter=10000,
                        random_state=int(random_state)
                    )
                    log_l2.fit(X_scaled.values, y)
                    if log_l2.coef_.ndim > 1:
                        coefs = np.abs(log_l2.coef_).mean(axis=0)
                    else:
                        coefs = np.abs(log_l2.coef_[0])

                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": coefs,
                    "Abs_Coefficient": np.abs(coefs)
                }).sort_values("Abs_Coefficient", ascending=False)

                # Ridge doesn't set coefficients to exactly zero
                # Select top features based on threshold
                threshold = st.sidebar.slider(
                    "Coefficient Threshold (percentile to keep)",
                    10, 100, 50, 5,
                    key="ridge_thresh"
                ) if False else 50  # Default threshold

                # Use top N% features
                n_keep = max(1, int(len(feature_cols) * 0.5))
                optimal_features = coef_df.head(n_keep)["Feature"].tolist()

                # Compute metric
                if task_type == "Classification":
                    eval_model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                else:
                    eval_model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                scores = cross_val_score(eval_model, X_scaled[optimal_features].values, y, cv=cv_folds, scoring=scoring)
                best_metric_val = scores.mean()

                history_df = coef_df

            st.success(f"✅ {fs_method} completed successfully!")

            # --- Results ---
            st.subheader("🏆 Selected Features")
            st.markdown(f"**Number of selected features:** {len(optimal_features)} out of {len(feature_cols)}")
            st.markdown(f"**Best CV {metric_name}:** {best_metric_val:.4f}")
            st.markdown(f"**Selected features:** `{', '.join(optimal_features)}`")

            # Selected vs Dropped
            dropped_features = [f for f in feature_cols if f not in optimal_features]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Selected Features")
                for i, f in enumerate(optimal_features, 1):
                    st.markdown(f"{i}. {f}")
            with col2:
                st.markdown("### ❌ Dropped Features")
                if dropped_features:
                    for i, f in enumerate(dropped_features, 1):
                        st.markdown(f"{i}. {f}")
                else:
                    st.markdown("None — all features were selected.")

            # --- Visualization ---
            st.subheader("📊 Feature Selection Visualization")

            if fs_method in ["Forward Selection", "Backward Elimination"]:
                # Step-by-step performance plot
                fig_steps = go.Figure()
                fig_steps.add_trace(go.Scatter(
                    x=history_df["Step"],
                    y=history_df[f"CV {metric_name}"],
                    mode='lines+markers',
                    marker=dict(size=10, color='blue'),
                    line=dict(color='blue', width=2),
                    name=f'CV {metric_name}'
                ))

                # Highlight best step
                fig_steps.add_trace(go.Scatter(
                    x=[history_df.iloc[best_step_idx]["Step"]],
                    y=[best_metric_val],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Best ({best_metric_val:.4f})'
                ))

                action = "Added" if fs_method == "Forward Selection" else "Removed"
                fig_steps.update_layout(
                    xaxis_title=f"Step (Feature {action})",
                    yaxis_title=f"CV {metric_name}",
                    title=f"{fs_method} - Performance at Each Step",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_steps, use_container_width=True)

                # History table
                st.subheader("📋 Selection History")
                st.dataframe(history_df, use_container_width=True)

            else:
                # Coefficient bar plot for Lasso / Ridge
                plot_df = coef_df.sort_values("Abs_Coefficient", ascending=True)

                # Color: selected vs not selected
                plot_df["Status"] = plot_df["Feature"].apply(
                    lambda x: "Selected" if x in optimal_features else "Dropped"
                )

                fig_coef = px.bar(
                    plot_df,
                    x="Coefficient",
                    y="Feature",
                    orientation='h',
                    color="Status",
                    color_discrete_map={"Selected": "green", "Dropped": "lightgray"},
                    title=f"{fs_method} - Feature Coefficients",
                    template="plotly_white"
                )
                fig_coef.update_layout(height=max(400, len(feature_cols) * 30))
                st.plotly_chart(fig_coef, use_container_width=True)

                # Coefficient table
                st.subheader("📋 Coefficient Details")
                display_df = coef_df[["Feature", "Coefficient", "Abs_Coefficient"]].copy()
                display_df["Selected"] = display_df["Feature"].apply(
                    lambda x: "✅" if x in optimal_features else "❌"
                )
                st.dataframe(display_df, use_container_width=True)

            # --- Comparison: All features vs Selected features ---
            st.subheader("📊 Performance Comparison")

            if task_type == "Classification":
                full_model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                sel_model = RandomForestClassifier(n_estimators=50, random_state=int(random_state), n_jobs=-1)
            else:
                full_model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)
                sel_model = RandomForestRegressor(n_estimators=50, random_state=int(random_state), n_jobs=-1)

            full_score = cross_val_score(full_model, X_scaled.values, y, cv=cv_folds, scoring=scoring).mean()

            if len(optimal_features) > 0:
                sel_score = cross_val_score(sel_model, X_scaled[optimal_features].values, y, cv=cv_folds, scoring=scoring).mean()
            else:
                sel_score = 0.0

            comp_df = pd.DataFrame({
                "Feature Set": ["All Features", "Selected Features"],
                f"CV {metric_name}": [full_score, sel_score],
                "N Features": [len(feature_cols), len(optimal_features)]
            })

            fig_comp = px.bar(
                comp_df,
                x="Feature Set",
                y=f"CV {metric_name}",
                text=comp_df[f"CV {metric_name}"].apply(lambda x: f"{x:.4f}"),
                color="Feature Set",
                color_discrete_map={"All Features": "steelblue", "Selected Features": "green"},
                title="All Features vs Selected Features Performance",
                template="plotly_white"
            )
            fig_comp.update_layout(height=450, showlegend=False)
            fig_comp.update_traces(textposition='outside')
            st.plotly_chart(fig_comp, use_container_width=True)

            st.table(comp_df)

            # --- Download ---
            st.subheader("📥 Download Results")
            result_csv = pd.DataFrame({
                "Feature": feature_cols,
                "Selected": ["Yes" if f in optimal_features else "No" for f in feature_cols]
            })
            csv = result_csv.to_csv(index=False)
            st.download_button(
                label="Download Feature Selection Results CSV",
                data=csv,
                file_name=f"feature_selection_{fs_method.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin Feature Selection.")
    st.subheader("📌 Supported Methods")
    st.markdown("""
    | Method | Description |
    |--------|-------------|
    | **Forward Selection** | Iteratively adds best feature one at a time |
    | **Backward Elimination** | Starts with all features, removes worst one at a time |
    | **Lasso (L1)** | Shrinks coefficients, sets some to exactly zero |
    | **Ridge (L2)** | Shrinks coefficients, ranks by importance |
    
    **Metrics displayed:** R² (Regression) or Accuracy (Classification), with cross-validation
    """)