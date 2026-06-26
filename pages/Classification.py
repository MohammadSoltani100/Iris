import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score
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
    roc_auc_score
)

st.set_page_config(page_title="Classification Analysis", layout="wide")
st.title("🏷️ Classification Analysis")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("⚙️ Classification Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Target variable (can be categorical)
    target_col = st.sidebar.selectbox("Select Target Variable (y)", options=all_columns)

    # Feature columns (numeric only)
    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Select Feature Variables (X)",
        options=available_features,
        default=available_features
    )

    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature variable.")
        st.stop()

    # Classification method
    clf_type = st.sidebar.selectbox(
        "Select Classification Method",
        [
            "Logistic Regression",
            "SVM (Support Vector Machine)",
            "Random Forest Classifier",
            "XGBoost Classifier"
        ]
    )

    # Method-specific parameters
    # Logistic Regression
    lr_C = 1.0
    lr_max_iter = 1000
    if clf_type == "Logistic Regression":
        lr_C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        lr_max_iter = st.sidebar.slider("Max Iterations", 100, 5000, 1000, 100)

    # SVM
    svm_C = 1.0
    svm_kernel = "rbf"
    if clf_type == "SVM (Support Vector Machine)":
        svm_C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        svm_kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])

    # Random Forest
    rf_n_est = 100
    rf_max_depth = None
    if clf_type == "Random Forest Classifier":
        rf_n_est = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
        rf_max_depth = st.sidebar.selectbox("Max Depth", [None, 3, 5, 10, 15, 20])

    # XGBoost
    xgb_lr = 0.1
    xgb_n_est = 100
    xgb_max_depth = 6
    if clf_type == "XGBoost Classifier":
        xgb_lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        xgb_n_est = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        xgb_max_depth = st.sidebar.slider("Max Depth", 2, 15, 6, 1)

    # General settings
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5) / 100
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    scale_features = st.sidebar.checkbox("Standardize Features", value=True)
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 2, 10, 5, 1)

    # --- Run Classification ---
    if st.sidebar.button("🚀 Run Classification", use_container_width=True):
        with st.spinner(f"Running {clf_type}..."):
            # Prepare data
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols].values
            y_raw = data[target_col].values

            # Encode target if necessary
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = le.classes_.astype(str)
            n_classes = len(class_names)

            # Scale features
            if scale_features:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=y
            )

            # Build model
            if clf_type == "Logistic Regression":
                model = LogisticRegression(
                    C=lr_C,
                    max_iter=lr_max_iter,
                    random_state=int(random_state),
                    multi_class='auto'
                )
            elif clf_type == "SVM (Support Vector Machine)":
                model = SVC(
                    C=svm_C,
                    kernel=svm_kernel,
                    random_state=int(random_state),
                    probability=True
                )
            elif clf_type == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=rf_n_est,
                    max_depth=rf_max_depth,
                    random_state=int(random_state),
                    n_jobs=-1
                )
            elif clf_type == "XGBoost Classifier":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        learning_rate=xgb_lr,
                        n_estimators=xgb_n_est,
                        max_depth=xgb_max_depth,
                        random_state=int(random_state),
                        n_jobs=-1,
                        verbosity=0,
                        use_label_encoder=False,
                        eval_metric='mlogloss' if n_classes > 2 else 'logloss'
                    )
                except ImportError:
                    st.error("❌ XGBoost is not installed.")
                    st.stop()

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probabilities (for ROC)
            try:
                y_test_proba = model.predict_proba(X_test)
            except Exception:
                y_test_proba = None

            # --- Metrics ---
            avg_method = 'weighted' if n_classes > 2 else 'binary'

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average=avg_method, zero_division=0)

            # Cross-validation
            if scale_features:
                X_cv = StandardScaler().fit_transform(data[feature_cols].values)
            else:
                X_cv = data[feature_cols].values

            if clf_type == "Logistic Regression":
                cv_model = LogisticRegression(C=lr_C, max_iter=lr_max_iter, random_state=int(random_state))
            elif clf_type == "SVM (Support Vector Machine)":
                cv_model = SVC(C=svm_C, kernel=svm_kernel, random_state=int(random_state))
            elif clf_type == "Random Forest Classifier":
                cv_model = RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_max_depth, random_state=int(random_state), n_jobs=-1)
            elif clf_type == "XGBoost Classifier":
                from xgboost import XGBClassifier
                cv_model = XGBClassifier(learning_rate=xgb_lr, n_estimators=xgb_n_est, max_depth=xgb_max_depth,
                                         random_state=int(random_state), n_jobs=-1, verbosity=0,
                                         use_label_encoder=False, eval_metric='mlogloss' if n_classes > 2 else 'logloss')

            cv_acc = cross_val_score(cv_model, X_cv, y, cv=cv_folds, scoring='accuracy').mean()
            cv_f1 = cross_val_score(cv_model, X_cv, y, cv=cv_folds, scoring='f1_weighted').mean()

            st.success(f"✅ {clf_type} completed successfully!")

            # --- Display Metrics ---
            st.subheader("📊 Performance Metrics")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Train Accuracy", f"{train_acc:.4f}")
            col2.metric("Test Accuracy", f"{test_acc:.4f}")
            col3.metric("Precision", f"{test_precision:.4f}")
            col4.metric("Recall", f"{test_recall:.4f}")
            col5.metric("F1-Score", f"{test_f1:.4f}")

            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Train": [f"{train_acc:.4f}", "—", "—", "—"],
                "Test": [f"{test_acc:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}", f"{test_f1:.4f}"],
                "CV Mean": [f"{cv_acc:.4f}", "—", "—", f"{cv_f1:.4f}"]
            })
            st.table(metrics_df)

            # --- Classification Report ---
            st.subheader("📝 Detailed Classification Report")
            report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

            # --- Confusion Matrix ---
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_test_pred)

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_names,
                y=class_names,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                aspect="auto"
            )
            fig_cm.update_layout(height=500, width=600, template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)

            # --- ROC Curve ---
            if y_test_proba is not None:
                st.subheader("📈 ROC Curve")

                if n_classes == 2:
                    # Binary ROC
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
                    roc_auc_val = auc(fpr, tpr)

                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc_val:.4f})',
                        line=dict(color='blue', width=2)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        title="ROC Curve",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    # Multi-class ROC (One vs Rest)
                    fig_roc = go.Figure()
                    for i, class_name in enumerate(class_names):
                        y_binary = (y_test == i).astype(int)
                        if y_binary.sum() > 0:
                            fpr, tpr, _ = roc_curve(y_binary, y_test_proba[:, i])
                            roc_auc_val = auc(fpr, tpr)
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'{class_name} (AUC = {roc_auc_val:.4f})'
                            ))

                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='grey', dash='dash')
                    ))
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        title="ROC Curves (One-vs-Rest)",
                        template="plotly_white",
                        height=550
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            # --- Feature Importance ---
            if clf_type in ["Random Forest Classifier", "XGBoost Classifier"]:
                st.subheader("🏆 Feature Importance")
                importances = model.feature_importances_

                imp_df = pd.DataFrame({
                    "Feature": feature_cols,
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
                fig_imp.update_layout(height=max(400, len(feature_cols) * 30))
                st.plotly_chart(fig_imp, use_container_width=True)

            # Coefficients for Logistic Regression
            if clf_type == "Logistic Regression":
                st.subheader("📐 Model Coefficients")
                if model.coef_.shape[0] == 1:
                    coef_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Coefficient": model.coef_[0]
                    }).sort_values("Coefficient", ascending=True)
                else:
                    # Multi-class: show coefficients for each class
                    coef_data = {}
                    for i, cn in enumerate(class_names):
                        coef_data[cn] = model.coef_[i]
                    coef_df = pd.DataFrame(coef_data, index=feature_cols)
                    st.dataframe(coef_df, use_container_width=True)
                    coef_df = None

                if coef_df is not None:
                    fig_coef = px.bar(
                        coef_df,
                        x="Coefficient",
                        y="Feature",
                        orientation='h',
                        title="Logistic Regression Coefficients",
                        template="plotly_white",
                        color="Coefficient",
                        color_continuous_scale="RdBu_r"
                    )
                    fig_coef.update_layout(height=max(400, len(feature_cols) * 30))
                    st.plotly_chart(fig_coef, use_container_width=True)

            # --- Download ---
            st.subheader("📥 Download Results")
            results_df = pd.DataFrame({
                "Actual": le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(y_test_pred)
            })
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name=f"classification_results_{clf_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin Classification Analysis.")
    st.subheader("📌 Supported Methods")
    st.markdown("""
    | Method | Description |
    |--------|-------------|
    | **Logistic Regression** | Linear classification model |
    | **SVM** | Support Vector Machine with various kernels |
    | **Random Forest** | Ensemble tree-based classifier |
    | **XGBoost** | Gradient boosting classifier |
    
    **Metrics displayed:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
    """)