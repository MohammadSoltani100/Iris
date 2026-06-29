"""
Machine Learning parent page with three tabs:
  1) Regression (Linear, Multi-Linear, Polynomial, Random Forest, XGBoost)
  2) Classification (Logistic, SVM, Random Forest, XGBoost)
  3) Feature Selection (Forward, Backward, Lasso, Ridge)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, PolynomialFeatures,
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Lasso, Ridge,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_data_widget, show_dataframe_overview, numeric_columns,
    download_dataframe, add_common_layout_options,
)

st.title("🤖 Machine Learning")
st.markdown("---")

tab_reg, tab_clf, tab_fs = st.tabs([
    "📈 Regression",
    "🏷️ Classification",
    "🎯 Feature Selection",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — REGRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_reg:
    st.subheader("Regression Analysis")
    st.write("Upload data with numeric target and feature columns.")

    df_r = load_data_widget("reg", "Upload regression data")
    if df_r is not None:
        show_dataframe_overview(df_r)
        num_r = numeric_columns(df_r)
        if len(num_r) < 2:
            st.error("Need at least 2 numeric columns.")
            st.stop()

        target_r = st.selectbox("Target Variable (y)", num_r, key="reg_y")
        avail_r = [c for c in num_r if c != target_r]
        feat_r = st.multiselect("Feature Variables (X)", avail_r,
                                default=avail_r, key="reg_x")
        if not feat_r:
            st.warning("Select at least one feature.")
            st.stop()

        reg_type = st.selectbox("Regression Method", [
            "Linear Regression", "Multi-Linear Regression",
            "Polynomial Regression", "Random Forest Regression",
            "XGBoost Regression",
        ], key="reg_meth")

        # Method-specific parameters
        poly_deg = 2
        if reg_type == "Polynomial Regression":
            poly_deg = st.slider("Polynomial Degree", 2, 5, 2,
                                 key="reg_poly")

        rf_n, rf_d = 100, None
        if reg_type == "Random Forest Regression":
            rf_n = st.slider("Number of Trees", 10, 500, 100, 10,
                             key="reg_rfn")
            rf_d = st.selectbox("Max Depth",
                                [None, 3, 5, 10, 15, 20],
                                key="reg_rfd")

        xg_lr, xg_n, xg_d = 0.1, 100, 6
        if reg_type == "XGBoost Regression":
            xg_lr = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01,
                              key="reg_xlr")
            xg_n = st.slider("N Estimators", 10, 500, 100, 10,
                             key="reg_xn")
            xg_d = st.slider("Max Depth", 2, 15, 6, key="reg_xd")

        # General settings
        st.markdown("---")
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            test_sz = st.slider("Test %", 10, 50, 20, 5,
                                key="reg_ts") / 100
        with gc2:
            rs = int(st.number_input("Random State", value=42,
                                     step=1, key="reg_rs"))
        with gc3:
            do_scale = st.checkbox("Standardize", True,
                                   key="reg_sc")
        with gc4:
            cv_k = st.slider("CV Folds", 2, 10, 5, key="reg_cv")

        if st.button("🚀 Run Regression", use_container_width=True,
                     key="reg_run"):

            # Resolve features for linear vs multi-linear
            feat_used = feat_r
            if reg_type == "Linear Regression" and len(feat_r) > 1:
                st.info("Linear Regression uses first feature only. "
                        "Use Multi-Linear for multiple.")
                feat_used = [feat_r[0]]
            elif (reg_type == "Multi-Linear Regression"
                  and len(feat_r) < 2):
                reg_type = "Linear Regression"
                feat_used = feat_r

            with st.spinner(f"Running {reg_type}..."):
                data_r = df_r[feat_used + [target_r]].dropna()
                X_raw = data_r[feat_used].values
                y = data_r[target_r].values

                # Scale
                scaler = (StandardScaler() if do_scale else None)
                X_sc = (scaler.fit_transform(X_raw) if scaler
                        else X_raw.copy())

                # Polynomial expansion
                poly_obj = None
                if reg_type == "Polynomial Regression":
                    poly_obj = PolynomialFeatures(
                        degree=poly_deg, include_bias=False)
                    X_sc = poly_obj.fit_transform(X_sc)

                # Split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sc, y, test_size=test_sz, random_state=rs)

                st.markdown("#### ✂️ Train / Test Split")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total", len(y))
                s2.metric("Train", len(y_tr),
                          f"{len(y_tr)/len(y)*100:.0f}%")
                s3.metric("Test", len(y_te),
                          f"{len(y_te)/len(y)*100:.0f}%")

                # Build model
                if reg_type in ["Linear Regression",
                                "Multi-Linear Regression",
                                "Polynomial Regression"]:
                    model = LinearRegression()
                elif reg_type == "Random Forest Regression":
                    model = RandomForestRegressor(
                        n_estimators=rf_n, max_depth=rf_d,
                        random_state=rs, n_jobs=-1)
                else:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        learning_rate=xg_lr, n_estimators=xg_n,
                        max_depth=xg_d, random_state=rs,
                        n_jobs=-1, verbosity=0)

                model.fit(X_tr, y_tr)
                y_tr_p = model.predict(X_tr)
                y_te_p = model.predict(X_te)

                # Metrics
                tr_r2 = r2_score(y_tr, y_tr_p)
                te_r2 = r2_score(y_te, y_te_p)
                tr_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_p))
                te_rmse = np.sqrt(mean_squared_error(y_te, y_te_p))
                tr_mae = mean_absolute_error(y_tr, y_tr_p)
                te_mae = mean_absolute_error(y_te, y_te_p)
                try:
                    te_mape = (mean_absolute_percentage_error(
                        y_te, y_te_p) * 100)
                except Exception:
                    te_mape = np.nan

                # Cross-validation with pipeline
                kf = KFold(n_splits=cv_k, shuffle=True,
                           random_state=rs)
                if reg_type == "Polynomial Regression":
                    cv_pipe = Pipeline([
                        ("sc", StandardScaler()),
                        ("poly", PolynomialFeatures(
                            degree=poly_deg, include_bias=False)),
                        ("reg", LinearRegression()),
                    ])
                else:
                    if reg_type in ["Linear Regression",
                                    "Multi-Linear Regression"]:
                        base = LinearRegression()
                    elif reg_type == "Random Forest Regression":
                        base = RandomForestRegressor(
                            n_estimators=rf_n, max_depth=rf_d,
                            random_state=rs, n_jobs=-1)
                    else:
                        from xgboost import XGBRegressor
                        base = XGBRegressor(
                            learning_rate=xg_lr,
                            n_estimators=xg_n,
                            max_depth=xg_d, random_state=rs,
                            n_jobs=-1, verbosity=0)
                    if do_scale:
                        cv_pipe = Pipeline([
                            ("sc", StandardScaler()),
                            ("reg", base)])
                    else:
                        cv_pipe = base

                cv_r2 = cross_val_score(cv_pipe, X_raw, y,
                                        cv=kf, scoring="r2")
                cv_mse = cross_val_score(
                    cv_pipe, X_raw, y, cv=kf,
                    scoring="neg_mean_squared_error")
                cv_rmse = np.sqrt(-cv_mse)

                st.success(f"✅ {reg_type} completed!")

                # Display metrics
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Train R²", f"{tr_r2:.4f}")
                m2.metric("Test R²", f"{te_r2:.4f}")
                m3.metric("Train RMSE", f"{tr_rmse:.4f}")
                m4.metric("Test RMSE", f"{te_rmse:.4f}")

                met_df = pd.DataFrame({
                    "Metric": ["R²", "RMSE", "MAE", "MAPE%"],
                    "Train": [f"{tr_r2:.4f}", f"{tr_rmse:.4f}",
                              f"{tr_mae:.4f}", "—"],
                    "Test": [f"{te_r2:.4f}", f"{te_rmse:.4f}",
                             f"{te_mae:.4f}",
                             f"{te_mape:.2f}"
                             if not np.isnan(te_mape) else "N/A"],
                    f"CV ({cv_k}-fold)": [
                        f"{cv_r2.mean():.4f}±{cv_r2.std():.4f}",
                        f"{cv_rmse.mean():.4f}±{cv_rmse.std():.4f}",
                        "—", "—"],
                })
                st.table(met_df)

                # CV fold chart
                cv_chart = pd.DataFrame({
                    "Fold": [f"F{i+1}" for i in range(cv_k)] * 2,
                    "Score": list(cv_r2) + list(cv_rmse),
                    "Metric": ["R²"] * cv_k + ["RMSE"] * cv_k,
                })
                fig_cv = px.bar(cv_chart, x="Fold", y="Score",
                                color="Metric", barmode="group",
                                title="CV Fold Performance")
                fig_cv = add_common_layout_options(fig_cv, height=400)
                st.plotly_chart(fig_cv, use_container_width=True)

                # Actual vs Predicted
                st.markdown("#### 📈 Actual vs Predicted")
                fig_avp = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Train", "Test"))
                fig_avp.add_trace(go.Scatter(
                    x=y_tr, y=y_tr_p, mode="markers",
                    marker=dict(color="blue", opacity=.6, size=7),
                    name="Train"), row=1, col=1)
                mn, mx = min(y_tr.min(), y_tr_p.min()), \
                    max(y_tr.max(), y_tr_p.max())
                fig_avp.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Perfect", showlegend=False),
                    row=1, col=1)
                fig_avp.add_trace(go.Scatter(
                    x=y_te, y=y_te_p, mode="markers",
                    marker=dict(color="green", opacity=.6, size=7),
                    name="Test"), row=1, col=2)
                mn2, mx2 = min(y_te.min(), y_te_p.min()), \
                    max(y_te.max(), y_te_p.max())
                fig_avp.add_trace(go.Scatter(
                    x=[mn2, mx2], y=[mn2, mx2], mode="lines",
                    line=dict(color="red", dash="dash"),
                    showlegend=False), row=1, col=2)
                fig_avp.update_layout(height=500,
                                      template="plotly_white")
                st.plotly_chart(fig_avp, use_container_width=True)

                # Residuals
                st.markdown("#### 📉 Residual Plot (Test)")
                resid = y_te - y_te_p
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=y_te_p, y=resid, mode="markers",
                    marker=dict(color="purple", opacity=.6)))
                fig_res.add_hline(y=0, line_dash="dash",
                                  line_color="red")
                fig_res.update_layout(
                    xaxis_title="Predicted",
                    yaxis_title="Residual",
                    height=450, template="plotly_white")
                st.plotly_chart(fig_res, use_container_width=True)

                # Feature importance / coefficients
                if reg_type in ["Random Forest Regression",
                                "XGBoost Regression"]:
                    st.markdown("#### 🏆 Feature Importance")
                    imp = pd.DataFrame({
                        "Feature": feat_used,
                        "Importance": model.feature_importances_,
                    }).sort_values("Importance", ascending=True)
                    fig_imp = px.bar(imp, x="Importance",
                                    y="Feature", orientation="h",
                                    color="Importance",
                                    color_continuous_scale="viridis")
                    fig_imp.update_layout(height=max(
                        400, len(feat_used)*30))
                    st.plotly_chart(fig_imp,
                                   use_container_width=True)

                elif reg_type in ["Linear Regression",
                                  "Multi-Linear Regression"]:
                    st.markdown("#### 📐 Coefficients")
                    coef_vals = (model.coef_ if model.coef_.ndim == 1
                                 else model.coef_.flatten())
                    cdf = pd.DataFrame({
                        "Feature": feat_used,
                        "Coefficient": coef_vals})
                    cdf = pd.concat([cdf, pd.DataFrame({
                        "Feature": ["Intercept"],
                        "Coefficient": [model.intercept_]})],
                        ignore_index=True)
                    st.table(cdf)

                # Download
                pred_df = pd.DataFrame({
                    "Actual": np.concatenate([y_tr, y_te]),
                    "Predicted": np.concatenate([y_tr_p, y_te_p]),
                    "Set": ["Train"]*len(y_tr) + ["Test"]*len(y_te),
                })
                download_dataframe(pred_df,
                                   f"regression_{reg_type}.csv",
                                   key="dl_reg")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_clf:
    st.subheader("Classification Analysis")
    st.write("Upload data with a categorical target and numeric features.")

    df_c = load_data_widget("clf", "Upload classification data")
    if df_c is not None:
        show_dataframe_overview(df_c)
        all_c = df_c.columns.tolist()
        num_c = numeric_columns(df_c)

        target_c = st.selectbox("Target Variable (y)", all_c,
                                key="clf_y")
        avail_c = [c for c in num_c if c != target_c]
        feat_c = st.multiselect("Feature Variables (X)", avail_c,
                                default=avail_c, key="clf_x")
        if not feat_c:
            st.warning("Select at least one feature.")
            st.stop()

        clf_type = st.selectbox("Classification Method", [
            "Logistic Regression", "SVM (Support Vector Machine)",
            "Random Forest Classifier", "XGBoost Classifier",
        ], key="clf_meth")

        # Method parameters
        lr_C, lr_mi = 1.0, 1000
        if clf_type == "Logistic Regression":
            lr_C = st.slider("C", 0.01, 10.0, 1.0, 0.01,
                             key="clf_c")
            lr_mi = st.slider("Max Iter", 100, 5000, 1000, 100,
                              key="clf_mi")

        sv_C, sv_k = 1.0, "rbf"
        if clf_type == "SVM (Support Vector Machine)":
            sv_C = st.slider("C", 0.01, 10.0, 1.0, 0.01,
                             key="clf_svc")
            sv_k = st.selectbox("Kernel",
                                ["rbf", "linear", "poly", "sigmoid"],
                                key="clf_svk")

        rf_cn, rf_cd = 100, None
        if clf_type == "Random Forest Classifier":
            rf_cn = st.slider("Trees", 10, 500, 100, 10,
                              key="clf_rfn")
            rf_cd = st.selectbox("Max Depth",
                                 [None, 3, 5, 10, 15, 20],
                                 key="clf_rfd")

        xc_lr, xc_n, xc_d = 0.1, 100, 6
        if clf_type == "XGBoost Classifier":
            xc_lr = st.slider("LR", 0.01, 0.5, 0.1, 0.01,
                              key="clf_xlr")
            xc_n = st.slider("N Est", 10, 500, 100, 10,
                             key="clf_xn")
            xc_d = st.slider("Depth", 2, 15, 6, key="clf_xd")

        st.markdown("---")
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            ts_c = st.slider("Test %", 10, 50, 20, 5,
                             key="clf_ts") / 100
        with gc2:
            rs_c = int(st.number_input("Random State", value=42,
                                       step=1, key="clf_rs"))
        with gc3:
            sc_c = st.checkbox("Standardize", True, key="clf_sc")
        with gc4:
            cv_c = st.slider("CV Folds", 2, 10, 5, key="clf_cv")

        if st.button("🚀 Run Classification",
                     use_container_width=True, key="clf_run"):
            with st.spinner(f"Running {clf_type}..."):
                data_c = df_c[feat_c + [target_c]].dropna()
                X_raw_c = data_c[feat_c].values
                y_raw_c = data_c[target_c].values

                le = LabelEncoder()
                y_c = le.fit_transform(y_raw_c)
                cnames = le.classes_.astype(str)
                nc = len(cnames)

                X_c = (StandardScaler().fit_transform(X_raw_c)
                       if sc_c else X_raw_c.copy())
                X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
                    X_c, y_c, test_size=ts_c,
                    random_state=rs_c, stratify=y_c)

                # Split info
                st.markdown("#### ✂️ Train / Test Split")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total", len(y_c))
                s2.metric("Train", len(y_tr_c))
                s3.metric("Test", len(y_te_c))

                # Class distribution chart
                tr_dist = pd.Series(y_tr_c).map(
                    dict(enumerate(cnames))).value_counts().reset_index()
                tr_dist.columns = ["Class", "Count"]
                tr_dist["Set"] = "Train"
                te_dist = pd.Series(y_te_c).map(
                    dict(enumerate(cnames))).value_counts().reset_index()
                te_dist.columns = ["Class", "Count"]
                te_dist["Set"] = "Test"
                dist_c = pd.concat([tr_dist, te_dist])
                fig_dist = px.bar(dist_c, x="Class", y="Count",
                                  color="Set", barmode="group",
                                  title="Class Distribution")
                fig_dist = add_common_layout_options(fig_dist,
                                                     height=380)
                st.plotly_chart(fig_dist, use_container_width=True)

                # Build model
                if clf_type == "Logistic Regression":
                    mdl = LogisticRegression(
                        C=lr_C, max_iter=lr_mi, random_state=rs_c)
                elif clf_type == "SVM (Support Vector Machine)":
                    mdl = SVC(C=sv_C, kernel=sv_k,
                              random_state=rs_c, probability=True)
                elif clf_type == "Random Forest Classifier":
                    mdl = RandomForestClassifier(
                        n_estimators=rf_cn, max_depth=rf_cd,
                        random_state=rs_c, n_jobs=-1)
                else:
                    from xgboost import XGBClassifier
                    mdl = XGBClassifier(
                        learning_rate=xc_lr, n_estimators=xc_n,
                        max_depth=xc_d, random_state=rs_c,
                        n_jobs=-1, verbosity=0,
                        use_label_encoder=False,
                        eval_metric="mlogloss" if nc > 2
                        else "logloss")

                mdl.fit(X_tr_c, y_tr_c)
                yp_tr = mdl.predict(X_tr_c)
                yp_te = mdl.predict(X_te_c)
                try:
                    yproba = mdl.predict_proba(X_te_c)
                except Exception:
                    yproba = None

                avg = "weighted" if nc > 2 else "binary"
                tr_acc = accuracy_score(y_tr_c, yp_tr)
                te_acc = accuracy_score(y_te_c, yp_te)
                te_prec = precision_score(y_te_c, yp_te,
                                          average=avg, zero_division=0)
                te_rec = recall_score(y_te_c, yp_te,
                                      average=avg, zero_division=0)
                te_f1 = f1_score(y_te_c, yp_te,
                                 average=avg, zero_division=0)

                # CV
                X_cv_c = (StandardScaler().fit_transform(X_raw_c)
                          if sc_c else X_raw_c.copy())
                skf = StratifiedKFold(n_splits=cv_c, shuffle=True,
                                      random_state=rs_c)

                if clf_type == "Logistic Regression":
                    cv_mdl = LogisticRegression(
                        C=lr_C, max_iter=lr_mi, random_state=rs_c)
                elif clf_type == "SVM (Support Vector Machine)":
                    cv_mdl = SVC(C=sv_C, kernel=sv_k,
                                 random_state=rs_c)
                elif clf_type == "Random Forest Classifier":
                    cv_mdl = RandomForestClassifier(
                        n_estimators=rf_cn, max_depth=rf_cd,
                        random_state=rs_c, n_jobs=-1)
                else:
                    from xgboost import XGBClassifier
                    cv_mdl = XGBClassifier(
                        learning_rate=xc_lr, n_estimators=xc_n,
                        max_depth=xc_d, random_state=rs_c,
                        n_jobs=-1, verbosity=0,
                        use_label_encoder=False,
                        eval_metric="mlogloss" if nc > 2
                        else "logloss")

                cv_acc = cross_val_score(cv_mdl, X_cv_c, y_c,
                                         cv=skf, scoring="accuracy")
                cv_f1s = cross_val_score(cv_mdl, X_cv_c, y_c,
                                          cv=skf,
                                          scoring="f1_weighted")

                st.success(f"✅ {clf_type} completed!")

                # Metrics display
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Train Acc", f"{tr_acc:.4f}")
                m2.metric("Test Acc", f"{te_acc:.4f}")
                m3.metric("Precision", f"{te_prec:.4f}")
                m4.metric("Recall", f"{te_rec:.4f}")
                m5.metric("F1", f"{te_f1:.4f}")

                mt_c = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision",
                               "Recall", "F1-Score"],
                    "Train": [f"{tr_acc:.4f}", "—", "—", "—"],
                    "Test": [f"{te_acc:.4f}", f"{te_prec:.4f}",
                             f"{te_rec:.4f}", f"{te_f1:.4f}"],
                    f"CV ({cv_c}-fold)": [
                        f"{cv_acc.mean():.4f}±{cv_acc.std():.4f}",
                        "—", "—",
                        f"{cv_f1s.mean():.4f}±{cv_f1s.std():.4f}"],
                })
                st.table(mt_c)

                # CV fold chart
                cv_cdf = pd.DataFrame({
                    "Fold": [f"F{i+1}" for i in range(cv_c)] * 2,
                    "Score": list(cv_acc) + list(cv_f1s),
                    "Metric": ["Accuracy"]*cv_c + ["F1"]*cv_c,
                })
                fig_cvc = px.bar(cv_cdf, x="Fold", y="Score",
                                 color="Metric", barmode="group",
                                 title="CV Performance")
                fig_cvc.update_layout(yaxis_range=[0, 1.05])
                st.plotly_chart(fig_cvc, use_container_width=True)

                # Classification report
                st.markdown("#### 📝 Classification Report (Test)")
                rpt = classification_report(
                    y_te_c, yp_te, target_names=cnames,
                    output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.style.format(
                    "{:.4f}"), use_container_width=True)

                # Confusion matrix
                st.markdown("#### 🔲 Confusion Matrix")
                cm = confusion_matrix(y_te_c, yp_te)
                fig_cm = px.imshow(cm, x=cnames, y=cnames,
                                   text_auto=True,
                                   color_continuous_scale="Blues",
                                   labels=dict(x="Predicted",
                                               y="Actual"))
                fig_cm.update_layout(height=500)
                st.plotly_chart(fig_cm, use_container_width=True)

                # ROC
                if yproba is not None:
                    st.markdown("#### 📈 ROC Curve")
                    fig_roc = go.Figure()
                    if nc == 2:
                        fpr, tpr, _ = roc_curve(y_te_c,
                                                yproba[:, 1])
                        a = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"AUC={a:.4f}"))
                    else:
                        for i, cn in enumerate(cnames):
                            yb = (y_te_c == i).astype(int)
                            if yb.sum() == 0:
                                continue
                            fpr, tpr, _ = roc_curve(yb,
                                                    yproba[:, i])
                            a = auc(fpr, tpr)
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr, mode="lines",
                                name=f"{cn} AUC={a:.4f}"))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dash", color="grey"),
                        name="Random"))
                    fig_roc.update_layout(
                        xaxis_title="FPR", yaxis_title="TPR",
                        height=500, template="plotly_white")
                    st.plotly_chart(fig_roc,
                                   use_container_width=True)

                # Feature importance
                if clf_type in ["Random Forest Classifier",
                                "XGBoost Classifier"]:
                    st.markdown("#### 🏆 Feature Importance")
                    imp_c = pd.DataFrame({
                        "Feature": feat_c,
                        "Importance": mdl.feature_importances_,
                    }).sort_values("Importance", ascending=True)
                    fig_ic = px.bar(imp_c, x="Importance",
                                   y="Feature", orientation="h",
                                   color="Importance",
                                   color_continuous_scale="viridis")
                    st.plotly_chart(fig_ic,
                                   use_container_width=True)

                download_dataframe(
                    pd.DataFrame({
                        "Actual": le.inverse_transform(y_te_c),
                        "Predicted": le.inverse_transform(yp_te)}),
                    f"classification_{clf_type}.csv",
                    key="dl_clf")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — FEATURE SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_fs:
    st.subheader("Feature Selection")
    st.write("Select the most informative features using various methods.")

    df_f = load_data_widget("fs", "Upload feature selection data")
    if df_f is not None:
        show_dataframe_overview(df_f)
        all_f = df_f.columns.tolist()
        num_f = numeric_columns(df_f)

        target_f = st.selectbox("Target (y)", all_f, key="fs_y")
        avail_f = [c for c in num_f if c != target_f]
        feat_f = st.multiselect("Candidate Features (X)", avail_f,
                                default=avail_f, key="fs_x")
        if len(feat_f) < 2:
            st.warning("Select at least 2 features.")
            st.stop()

        task = st.radio("Task Type",
                        ["Regression", "Classification"],
                        key="fs_task")
        fs_method = st.selectbox("Method", [
            "Forward Selection", "Backward Elimination",
            "Lasso (L1)", "Ridge (L2)",
        ], key="fs_meth")

        alpha_f = 1.0
        if fs_method in ["Lasso (L1)", "Ridge (L2)"]:
            alpha_f = st.slider("Alpha", 0.001, 10.0, 1.0, 0.001,
                                key="fs_alpha")

        cv_f = st.slider("CV Folds", 2, 10, 5, key="fs_cv")
        rs_f = int(st.number_input("Random State", value=42,
                                   step=1, key="fs_rs"))
        sc_f = st.checkbox("Standardize", True, key="fs_sc")

        if st.button("🚀 Run Feature Selection",
                     use_container_width=True, key="fs_run"):
            with st.spinner(f"Running {fs_method}..."):
                data_f = df_f[feat_f + [target_f]].dropna()
                X_f = data_f[feat_f]
                y_raw_f = data_f[target_f]

                if task == "Classification":
                    le_f = LabelEncoder()
                    y_f = le_f.fit_transform(y_raw_f)
                    scoring = "accuracy"
                    mname = "Accuracy"
                else:
                    y_f = y_raw_f.values.astype(float)
                    scoring = "r2"
                    mname = "R²"

                X_sc_f = pd.DataFrame(
                    StandardScaler().fit_transform(X_f),
                    columns=feat_f, index=X_f.index
                ) if sc_f else X_f.copy()

                def _get_eval_model():
                    if task == "Classification":
                        return RandomForestClassifier(
                            n_estimators=50, random_state=rs_f,
                            n_jobs=-1)
                    return RandomForestRegressor(
                        n_estimators=50, random_state=rs_f,
                        n_jobs=-1)

                # ── Forward Selection ──
                if fs_method == "Forward Selection":
                    selected, remaining = [], list(feat_f)
                    history = []
                    for step in range(len(feat_f)):
                        best_score, best_feat = -np.inf, None
                        for f in remaining:
                            cur = selected + [f]
                            sc_arr = cross_val_score(
                                _get_eval_model(),
                                X_sc_f[cur].values, y_f,
                                cv=cv_f, scoring=scoring)
                            if sc_arr.mean() > best_score:
                                best_score = sc_arr.mean()
                                best_feat = f
                        if best_feat:
                            selected.append(best_feat)
                            remaining.remove(best_feat)
                            history.append({
                                "Step": step+1,
                                "Added": best_feat,
                                f"CV {mname}": best_score})
                    hist_df = pd.DataFrame(history)
                    best_idx = hist_df[f"CV {mname}"].idxmax()
                    optimal = selected[:best_idx+1]
                    best_val = hist_df[f"CV {mname}"].max()

                # ── Backward Elimination ──
                elif fs_method == "Backward Elimination":
                    selected = list(feat_f)
                    history = []
                    sc0 = cross_val_score(
                        _get_eval_model(),
                        X_sc_f[selected].values, y_f,
                        cv=cv_f, scoring=scoring).mean()
                    history.append({"Step": 0,
                                    "Removed": "None",
                                    f"CV {mname}": sc0})
                    for step in range(len(feat_f)-1):
                        best_score, worst = -np.inf, None
                        for f in selected:
                            cur = [x for x in selected if x != f]
                            if not cur:
                                continue
                            sc_arr = cross_val_score(
                                _get_eval_model(),
                                X_sc_f[cur].values, y_f,
                                cv=cv_f, scoring=scoring)
                            if sc_arr.mean() > best_score:
                                best_score = sc_arr.mean()
                                worst = f
                        if worst:
                            selected.remove(worst)
                            history.append({
                                "Step": step+1,
                                "Removed": worst,
                                f"CV {mname}": best_score})
                    hist_df = pd.DataFrame(history)
                    best_idx = hist_df[f"CV {mname}"].idxmax()
                    all_copy = list(feat_f)
                    for i in range(1, best_idx+1):
                        rm = hist_df.iloc[i]["Removed"]
                        if rm in all_copy:
                            all_copy.remove(rm)
                    optimal = all_copy
                    best_val = hist_df[f"CV {mname}"].max()

                # ── Lasso ──
                elif fs_method == "Lasso (L1)":
                    if task == "Regression":
                        mdl_l = Lasso(alpha=alpha_f,
                                      random_state=rs_f,
                                      max_iter=10000)
                        mdl_l.fit(X_sc_f.values, y_f)
                        coefs = mdl_l.coef_
                    else:
                        mdl_l = LogisticRegression(
                            penalty="l1", C=1/alpha_f,
                            solver="saga", max_iter=10000,
                            random_state=rs_f)
                        mdl_l.fit(X_sc_f.values, y_f)
                        coefs = (np.abs(mdl_l.coef_).mean(axis=0)
                                 if mdl_l.coef_.ndim > 1
                                 else np.abs(mdl_l.coef_[0]))
                    coef_df = pd.DataFrame({
                        "Feature": feat_f,
                        "Coefficient": coefs,
                        "Abs": np.abs(coefs),
                    }).sort_values("Abs", ascending=False)
                    optimal = coef_df[
                        coef_df["Abs"] > 1e-6]["Feature"].tolist()
                    hist_df = coef_df
                    if optimal:
                        best_val = cross_val_score(
                            _get_eval_model(),
                            X_sc_f[optimal].values, y_f,
                            cv=cv_f, scoring=scoring).mean()
                    else:
                        best_val = 0.0

                # ── Ridge ──
                else:
                    if task == "Regression":
                        mdl_l = Ridge(alpha=alpha_f)
                        mdl_l.fit(X_sc_f.values, y_f)
                        coefs = mdl_l.coef_
                    else:
                        mdl_l = LogisticRegression(
                            penalty="l2", C=1/alpha_f,
                            solver="lbfgs", max_iter=10000,
                            random_state=rs_f)
                        mdl_l.fit(X_sc_f.values, y_f)
                        coefs = (np.abs(mdl_l.coef_).mean(axis=0)
                                 if mdl_l.coef_.ndim > 1
                                 else np.abs(mdl_l.coef_[0]))
                    coef_df = pd.DataFrame({
                        "Feature": feat_f,
                        "Coefficient": coefs,
                        "Abs": np.abs(coefs),
                    }).sort_values("Abs", ascending=False)
                    n_keep = max(1, len(feat_f) // 2)
                    optimal = coef_df.head(n_keep)[
                        "Feature"].tolist()
                    hist_df = coef_df
                    best_val = cross_val_score(
                        _get_eval_model(),
                        X_sc_f[optimal].values, y_f,
                        cv=cv_f, scoring=scoring).mean()

                st.success(f"✅ {fs_method} completed!")

                # Results
                dropped = [f for f in feat_f if f not in optimal]
                st.markdown(f"**Selected:** {len(optimal)} / "
                            f"{len(feat_f)} features")
                st.markdown(f"**Best CV {mname}:** "
                            f"{best_val:.4f}")

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("### ✅ Selected")
                    for i, f in enumerate(optimal, 1):
                        st.write(f"{i}. {f}")
                with rc2:
                    st.markdown("### ❌ Dropped")
                    for i, f in enumerate(dropped, 1):
                        st.write(f"{i}. {f}")

                # Visualization
                if fs_method in ["Forward Selection",
                                 "Backward Elimination"]:
                    fig_fs = go.Figure()
                    fig_fs.add_trace(go.Scatter(
                        x=hist_df["Step"],
                        y=hist_df[f"CV {mname}"],
                        mode="lines+markers",
                        marker=dict(size=10)))
                    fig_fs.add_trace(go.Scatter(
                        x=[hist_df.iloc[best_idx]["Step"]],
                        y=[best_val],
                        mode="markers",
                        marker=dict(size=15, color="red",
                                    symbol="star"),
                        name="Best"))
                    fig_fs.update_layout(
                        xaxis_title="Step",
                        yaxis_title=f"CV {mname}",
                        height=500, template="plotly_white",
                        title=f"{fs_method} Performance")
                    st.plotly_chart(fig_fs,
                                   use_container_width=True)
                    st.dataframe(hist_df,
                                 use_container_width=True)
                else:
                    plot_coef = coef_df.sort_values(
                        "Abs", ascending=True).copy()
                    plot_coef["Status"] = plot_coef[
                        "Feature"].apply(
                        lambda x: "Selected"
                        if x in optimal else "Dropped")
                    fig_fs = px.bar(
                        plot_coef, x="Coefficient",
                        y="Feature", orientation="h",
                        color="Status",
                        color_discrete_map={
                            "Selected": "green",
                            "Dropped": "lightgray"},
                        title=f"{fs_method} Coefficients")
                    fig_fs.update_layout(
                        height=max(400, len(feat_f)*30))
                    st.plotly_chart(fig_fs,
                                   use_container_width=True)

                # Comparison
                st.markdown("#### All vs Selected Performance")
                full_sc = cross_val_score(
                    _get_eval_model(), X_sc_f.values, y_f,
                    cv=cv_f, scoring=scoring).mean()
                comp = pd.DataFrame({
                    "Set": ["All Features",
                            "Selected Features"],
                    f"CV {mname}": [full_sc, best_val],
                    "N": [len(feat_f), len(optimal)],
                })
                fig_comp = px.bar(comp, x="Set",
                                  y=f"CV {mname}",
                                  text=comp[f"CV {mname}"].apply(
                                      lambda x: f"{x:.4f}"),
                                  color="Set")
                fig_comp.update_traces(textposition="outside")
                fig_comp.update_layout(height=400)
                st.plotly_chart(fig_comp,
                                use_container_width=True)

                download_dataframe(
                    pd.DataFrame({
                        "Feature": feat_f,
                        "Selected": ["Yes" if f in optimal
                                     else "No"
                                     for f in feat_f]}),
                    f"feature_selection_{fs_method}.csv",
                    key="dl_fs")