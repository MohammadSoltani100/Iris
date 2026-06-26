import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page title (set_page_config is in app.py)
# ─────────────────────────────────────────────
st.title("🏷️ Classification Analysis")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar settings
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Classification Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # ── Load data ─────────────────────────────
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Show basic dataset info
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Rows", df.shape[0])
    col_info2.metric("Total Columns", df.shape[1])
    col_info3.metric("Missing Values", df.isnull().sum().sum())

    all_columns  = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Target variable ───────────────────────
    target_col = st.sidebar.selectbox(
        "Select Target Variable (y)", options=all_columns
    )

    # ── Feature columns ───────────────────────
    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Select Feature Variables (X)",
        options=available_features,
        default=available_features,
    )

    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature variable.")
        st.stop()

    # ── Classification method ─────────────────
    clf_type = st.sidebar.selectbox(
        "Select Classification Method",
        [
            "Logistic Regression",
            "SVM (Support Vector Machine)",
            "Random Forest Classifier",
            "XGBoost Classifier",
        ],
    )

    # ── Method-specific hyper-parameters ──────

    # Logistic Regression parameters
    lr_C, lr_max_iter = 1.0, 1000
    if clf_type == "Logistic Regression":
        lr_C        = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        lr_max_iter = st.sidebar.slider("Max Iterations", 100, 5000, 1000, 100)

    # SVM parameters
    svm_C, svm_kernel = 1.0, "rbf"
    if clf_type == "SVM (Support Vector Machine)":
        svm_C      = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        svm_kernel = st.sidebar.selectbox(
            "Kernel", ["rbf", "linear", "poly", "sigmoid"]
        )

    # Random Forest parameters
    rf_n_est, rf_max_depth = 100, None
    if clf_type == "Random Forest Classifier":
        rf_n_est   = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
        rf_max_depth = st.sidebar.selectbox(
            "Max Depth", [None, 3, 5, 10, 15, 20]
        )

    # XGBoost parameters
    xgb_lr, xgb_n_est, xgb_max_depth = 0.1, 100, 6
    if clf_type == "XGBoost Classifier":
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
    run_btn = st.sidebar.button("🚀 Run Classification", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running {clf_type} …"):

            # ── Prepare data ───────────────────────────
            data   = df[feature_cols + [target_col]].dropna()
            X_raw  = data[feature_cols].values
            y_raw  = data[target_col].values

            # Encode labels
            le           = LabelEncoder()
            y            = le.fit_transform(y_raw)
            class_names  = le.classes_.astype(str)
            n_classes    = len(class_names)

            # ── Scale features ─────────────────────────
            scaler = StandardScaler() if scale_data else None
            X = scaler.fit_transform(X_raw) if scaler else X_raw.copy()

            # ── Train / Test split ─────────────────────
            # stratify=y ensures class distribution is preserved in both sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,       # keep class proportions
            )

            # Show split sizes
            st.subheader("✂️ Train / Test Split")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples",  len(y))
            c2.metric("Training Samples", len(y_train),
                      f"{len(y_train)/len(y)*100:.1f}%")
            c3.metric("Test Samples",   len(y_test),
                      f"{len(y_test)/len(y)*100:.1f}%")

            # Show class distribution in train vs test
            train_dist = pd.Series(y_train).map(
                dict(enumerate(class_names))
            ).value_counts().reset_index()
            train_dist.columns = ["Class", "Count"]
            train_dist["Set"] = "Train"

            test_dist = pd.Series(y_test).map(
                dict(enumerate(class_names))
            ).value_counts().reset_index()
            test_dist.columns = ["Class", "Count"]
            test_dist["Set"] = "Test"

            dist_df = pd.concat([train_dist, test_dist], ignore_index=True)

            fig_dist = px.bar(
                dist_df, x="Class", y="Count", color="Set",
                barmode="group",
                title="Class Distribution — Train vs Test",
                template="plotly_white",
                color_discrete_map={"Train": "#1E88E5", "Test": "#43A047"},
            )
            fig_dist.update_layout(height=380)
            st.plotly_chart(fig_dist, use_container_width=True)

            # ── Build model ────────────────────────────
            if clf_type == "Logistic Regression":
                model = LogisticRegression(
                    C=lr_C, max_iter=lr_max_iter,
                    random_state=random_state, multi_class="auto",
                )
            elif clf_type == "SVM (Support Vector Machine)":
                model = SVC(
                    C=svm_C, kernel=svm_kernel,
                    random_state=random_state, probability=True,
                )
            elif clf_type == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=rf_n_est, max_depth=rf_max_depth,
                    random_state=random_state, n_jobs=-1,
                )
            elif clf_type == "XGBoost Classifier":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=random_state,
                        n_jobs=-1, verbosity=0,
                        use_label_encoder=False,
                        eval_metric="mlogloss" if n_classes > 2 else "logloss",
                    )
                except ImportError:
                    st.error("❌ XGBoost is not installed.")
                    st.stop()

            # ── Train on training set ──────────────────
            model.fit(X_train, y_train)

            # ── Predict ────────────────────────────────
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            # Probabilities for ROC
            try:
                y_test_proba = model.predict_proba(X_test)
            except Exception:
                y_test_proba = None

            # ── Evaluate metrics ───────────────────────
            avg = "weighted" if n_classes > 2 else "binary"

            train_acc  = accuracy_score(y_train, y_train_pred)
            test_acc   = accuracy_score(y_test,  y_test_pred)
            test_prec  = precision_score(y_test, y_test_pred, average=avg, zero_division=0)
            test_rec   = recall_score(y_test,    y_test_pred, average=avg, zero_division=0)
            test_f1    = f1_score(y_test,        y_test_pred, average=avg, zero_division=0)

            # ── Cross-Validation on full dataset ───────
            # Re-scale the full dataset for CV
            X_cv = StandardScaler().fit_transform(X_raw) if scale_data else X_raw.copy()
            skf  = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=random_state)

            # Build an identical model for CV (fresh instance)
            if clf_type == "Logistic Regression":
                cv_model = LogisticRegression(
                    C=lr_C, max_iter=lr_max_iter, random_state=random_state,
                    multi_class="auto",
                )
            elif clf_type == "SVM (Support Vector Machine)":
                cv_model = SVC(
                    C=svm_C, kernel=svm_kernel,
                    random_state=random_state, probability=True,
                )
            elif clf_type == "Random Forest Classifier":
                cv_model = RandomForestClassifier(
                    n_estimators=rf_n_est, max_depth=rf_max_depth,
                    random_state=random_state, n_jobs=-1,
                )
            elif clf_type == "XGBoost Classifier":
                from xgboost import XGBClassifier
                cv_model = XGBClassifier(
                    learning_rate=xgb_lr, n_estimators=xgb_n_est,
                    max_depth=xgb_max_depth, random_state=random_state,
                    n_jobs=-1, verbosity=0, use_label_encoder=False,
                    eval_metric="mlogloss" if n_classes > 2 else "logloss",
                )

            cv_acc_scores = cross_val_score(
                cv_model, X_cv, y, cv=skf, scoring="accuracy"
            )
            cv_f1_scores  = cross_val_score(
                cv_model, X_cv, y, cv=skf, scoring="f1_weighted"
            )
            cv_acc_mean, cv_acc_std = cv_acc_scores.mean(), cv_acc_scores.std()
            cv_f1_mean,  cv_f1_std  = cv_f1_scores.mean(),  cv_f1_scores.std()

            st.success(f"✅ {clf_type} completed successfully!")

            # ─────────────────────────────────────────────
            # Section 1 — Performance Metrics
            # ─────────────────────────────────────────────
            st.subheader("📊 Performance Metrics")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Train Accuracy",  f"{train_acc:.4f}")
            m2.metric("Test Accuracy",   f"{test_acc:.4f}")
            m3.metric("Precision",       f"{test_prec:.4f}")
            m4.metric("Recall",          f"{test_rec:.4f}")
            m5.metric("F1-Score",        f"{test_f1:.4f}")

            metrics_table = pd.DataFrame({
                "Metric":   ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Train":    [f"{train_acc:.4f}", "—", "—", "—"],
                "Test":     [f"{test_acc:.4f}",
                             f"{test_prec:.4f}",
                             f"{test_rec:.4f}",
                             f"{test_f1:.4f}"],
                f"CV Mean ({cv_folds}-fold)": [
                    f"{cv_acc_mean:.4f} ± {cv_acc_std:.4f}",
                    "—", "—",
                    f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}",
                ],
            })
            st.table(metrics_table)

            # ── CV fold-by-fold bar chart ──────────────
            st.subheader(f"📉 Cross-Validation Results ({cv_folds} Folds)")

            cv_df = pd.DataFrame({
                "Fold":     [f"Fold {i+1}" for i in range(cv_folds)] * 2,
                "Score":    list(cv_acc_scores) + list(cv_f1_scores),
                "Metric":   ["Accuracy"] * cv_folds + ["F1-Score"] * cv_folds,
            })
            fig_cv = px.bar(
                cv_df, x="Fold", y="Score", color="Metric",
                barmode="group",
                title=f"CV Performance per Fold — {clf_type}",
                template="plotly_white",
                color_discrete_map={"Accuracy": "#1E88E5", "F1-Score": "#43A047"},
            )
            fig_cv.update_layout(height=420, yaxis_range=[0, 1.05])
            st.plotly_chart(fig_cv, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 2 — Detailed Classification Report
            # ─────────────────────────────────────────────
            st.subheader("📝 Detailed Classification Report (Test Set)")
            report    = classification_report(
                y_test, y_test_pred, target_names=class_names, output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.style.format("{:.4f}"), use_container_width=True
            )

            # ─────────────────────────────────────────────
            # Section 3 — Confusion Matrix
            # ─────────────────────────────────────────────
            st.subheader("🔲 Confusion Matrix (Test Set)")
            cm = confusion_matrix(y_test, y_test_pred)

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_names, y=class_names,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                aspect="auto",
            )
            fig_cm.update_layout(
                height=500, width=600, template="plotly_white"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 4 — ROC Curve
            # ─────────────────────────────────────────────
            if y_test_proba is not None:
                st.subheader("📈 ROC Curve (Test Set)")

                if n_classes == 2:
                    # Binary ROC
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
                    roc_auc_val = auc(fpr, tpr)

                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"AUC = {roc_auc_val:.4f}",
                        line=dict(color="blue", width=2),
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random", line=dict(color="red", dash="dash"),
                    ))
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        title=f"ROC Curve (AUC = {roc_auc_val:.4f})",
                        template="plotly_white", height=500,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                else:
                    # Multi-class One-vs-Rest ROC
                    fig_roc = go.Figure()
                    for i, cname in enumerate(class_names):
                        y_bin = (y_test == i).astype(int)
                        if y_bin.sum() == 0:
                            continue
                        fpr, tpr, _ = roc_curve(y_bin, y_test_proba[:, i])
                        roc_auc_val = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{cname} (AUC={roc_auc_val:.4f})",
                        ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random", line=dict(color="grey", dash="dash"),
                    ))
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        title="ROC Curves — One-vs-Rest",
                        template="plotly_white", height=550,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            # ─────────────────────────────────────────────
            # Section 5 — Feature Importance
            # ─────────────────────────────────────────────
            if clf_type in ["Random Forest Classifier", "XGBoost Classifier"]:
                st.subheader("🏆 Feature Importance")
                imp_df = pd.DataFrame({
                    "Feature":    feature_cols,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=True)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    title="Feature Importance",
                    template="plotly_white",
                    color="Importance",
                    color_continuous_scale="viridis",
                )
                fig_imp.update_layout(height=max(400, len(feature_cols) * 30))
                st.plotly_chart(fig_imp, use_container_width=True)

            # Logistic Regression coefficients
            if clf_type == "Logistic Regression":
                st.subheader("📐 Model Coefficients")
                if model.coef_.shape[0] == 1:
                    coef_df = pd.DataFrame({
                        "Feature":     feature_cols,
                        "Coefficient": model.coef_[0],
                    }).sort_values("Coefficient", ascending=True)
                    fig_coef = px.bar(
                        coef_df, x="Coefficient", y="Feature",
                        orientation="h",
                        title="Logistic Regression Coefficients",
                        template="plotly_white",
                        color="Coefficient",
                        color_continuous_scale="RdBu_r",
                    )
                    fig_coef.update_layout(
                        height=max(400, len(feature_cols) * 30)
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)
                else:
                    # Multi-class coefficients table
                    coef_data = {
                        cn: model.coef_[i]
                        for i, cn in enumerate(class_names)
                    }
                    st.dataframe(
                        pd.DataFrame(coef_data, index=feature_cols),
                        use_container_width=True,
                    )

            # ─────────────────────────────────────────────
            # Section 6 — Download
            # ─────────────────────────────────────────────
            st.subheader("📥 Download Results")
            results_df = pd.DataFrame({
                "Actual":    le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(y_test_pred),
                "Correct":   y_test == y_test_pred,
            })
            st.download_button(
                label="Download Test Predictions CSV",
                data=results_df.to_csv(index=False),
                file_name=f"classification_{clf_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # ── Info when no file uploaded ─────────────
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    st.subheader("📌 Supported Methods")
    st.markdown("""
| Method | Description |
|--------|-------------|
| **Logistic Regression** | Linear probabilistic classifier |
| **SVM** | Support Vector Machine with various kernels |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting classifier |

**Metrics:** Accuracy · Precision · Recall · F1-Score · ROC-AUC · Confusion Matrix  
**Extra:** Train/Test split with class distribution chart · CV fold-by-fold chart
""")