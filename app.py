"""
Heart Attack Prediction - Streamlit App
=======================================
Run locally:   streamlit run app.py

Needs (in the same folder):
  - heart_attack_model.joblib   (created by heart_attack_analysis.py)
  - model_metadata.json         (created by heart_attack_analysis.py)
  - figures/                    (optional - shown in the "Model Performance" tab)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_attack_model.joblib"
META_PATH = BASE_DIR / "model_metadata.json"
FIG_DIR = BASE_DIR / "figures"

st.set_page_config(page_title="Heart Attack Predictor", page_icon="🫀", layout="wide")


# --------------------------------------------------------------------------
# Load model (cached so it is only loaded once)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python heart_attack_analysis.py` first "
             "so that `heart_attack_model.joblib` and `model_metadata.json` are created.")
    st.stop()

ranges = meta["ranges"]

# Feature names in the same order as the model's transformed columns
CONTRIB_NAMES = ["log(CK-MB)", "log(Troponin)", "Age", "Heart rate",
                 "Systolic BP", "Diastolic BP", "Glucose", "Gender"]

# --------------------------------------------------------------------------
# Example patients (real rows from the dataset)
# --------------------------------------------------------------------------
PRESETS = {
    "Custom (normal-range starting values)": {
        "age": 55, "gender": "Male", "heart_rate": 72, "systolic_bp": 120,
        "diastolic_bp": 80, "glucose": 100.0, "ck_mb": 1.5, "troponin": 0.005,
    },
    "Example: patient with high cardiac markers": {
        "age": 64, "gender": "Male", "heart_rate": 70, "systolic_bp": 120,
        "diastolic_bp": 55, "glucose": 270.0, "ck_mb": 13.87, "troponin": 0.122,
    },
    "Example: patient with normal cardiac markers": {
        "age": 64, "gender": "Male", "heart_rate": 66, "systolic_bp": 160,
        "diastolic_bp": 83, "glucose": 160.0, "ck_mb": 1.80, "troponin": 0.012,
    },
}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🫀 Heart Attack Predictor")
    st.caption("Logistic Regression model")
    st.markdown("---")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="Probability above which a patient is classified as 'positive'. "
             "Lower it to catch more heart attacks (higher recall, more false alarms); "
             "raise it to reduce false alarms.")
    st.markdown("---")
    st.markdown("**Model quality (held-out test set)**")
    m = meta["test_metrics_threshold_0.5"]
    st.metric("ROC-AUC", f"{meta['test_auc']:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['recall_sensitivity']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c1.metric("F1", f"{m['f1']:.3f}")
    c2.metric("Accuracy", f"{m['accuracy']:.3f}")
    st.markdown("---")
    st.warning("⚠️ Educational project only. **Not a medical device** and not a "
               "substitute for professional diagnosis.")

tab_predict, tab_perf, tab_about = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ==========================================================================
# TAB 1 - PREDICT
# ==========================================================================
with tab_predict:
    st.header("Patient details")
    preset_name = st.selectbox("Quick fill", list(PRESETS.keys()))
    p = PRESETS[preset_name]
    k = preset_name  # changing preset changes widget keys -> resets defaults

    with st.form("patient_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age (years)", 1, 120, p["age"], key=f"age_{k}")
        gender = c2.selectbox("Gender", ["Male", "Female"],
                              index=0 if p["gender"] == "Male" else 1, key=f"g_{k}")
        heart_rate = c3.number_input("Heart rate (bpm)", 20, 220, p["heart_rate"], key=f"hr_{k}")
        glucose = c4.number_input("Blood glucose (mg/dL)", 30.0, 700.0, float(p["glucose"]),
                                  step=1.0, key=f"glu_{k}")

        c5, c6, c7, c8 = st.columns(4)
        sys_bp = c5.number_input("Systolic BP (mmHg)", 40, 260, p["systolic_bp"], key=f"sys_{k}")
        dia_bp = c6.number_input("Diastolic BP (mmHg)", 20, 180, p["diastolic_bp"], key=f"dia_{k}")
        ck_mb = c7.number_input("CK-MB (ng/mL)", 0.1, 500.0, float(p["ck_mb"]),
                                step=0.1, format="%.2f", key=f"ck_{k}")
        troponin = c8.number_input("Troponin (ng/mL)", 0.001, 20.0, float(p["troponin"]),
                                   step=0.001, format="%.3f", key=f"tro_{k}")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "heart_rate": heart_rate,
            "systolic_bp": sys_bp,
            "diastolic_bp": dia_bp,
            "glucose": glucose,
            "ck_mb": ck_mb,
            "troponin": troponin,
        }])[meta["features"]]

        prob = float(model.predict_proba(row)[0, 1])
        is_positive = prob >= threshold
        prob_txt = ">99.9%" if prob > 0.999 else ("<0.1%" if prob < 0.001 else f"{prob:.1%}")

        st.markdown("---")
        left, right = st.columns([1, 1.2])

        with left:
            st.subheader("Result")
            if is_positive:
                st.error(f"### 🚨 Heart attack risk: **POSITIVE**\nProbability: **{prob_txt}**")
            else:
                st.success(f"### ✅ Heart attack risk: **NEGATIVE**\nProbability: **{prob_txt}**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Classified using a decision threshold of {threshold:.2f}.")

            # Warn if any input is outside the range the model was trained on
            out_of_range = [
                name for name, val in row.iloc[0].items()
                if name != "gender" and not (ranges[name]["min"] <= val <= ranges[name]["max"])
            ]
            if out_of_range:
                st.warning("Some values are outside the range seen during training "
                           f"({', '.join(out_of_range)}). The prediction may be less reliable.")

        with right:
            st.subheader("Why this prediction?")
            prep = model.named_steps["prep"]
            clf = model.named_steps["clf"]
            transformed = prep.transform(row)[0]
            contrib = transformed * clf.coef_.ravel()
            # transformed column order: ck_mb, troponin, age, heart_rate, systolic, diastolic, glucose, gender
            df_c = pd.DataFrame({"feature": CONTRIB_NAMES, "contribution": contrib})
            df_c = df_c.sort_values("contribution")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(df_c["feature"], df_c["contribution"],
                    color=["#E4572E" if v > 0 else "#2E86AB" for v in df_c["contribution"]])
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("Contribution to log-odds  (red = pushes toward positive)")
            ax.set_title(f"Intercept = {clf.intercept_[0]:.2f} (added to the sum of bars)", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Contribution = (standardised feature value) x (model coefficient). "
                       "The model relies mostly on troponin and CK-MB.")

        with st.expander("Show the input row sent to the model"):
            st.dataframe(row)

    # ---------------- Batch prediction ----------------
    st.markdown("---")
    st.header("Batch prediction (CSV upload)")
    st.caption("Upload a CSV with columns: age, gender (1=male, 0=female), impluse *or* heart_rate, "
               "pressurehight *or* systolic_bp, pressurelow *or* diastolic_bp, glucose, kcm *or* ck_mb, troponin.")
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        raw = raw.rename(columns={"impluse": "heart_rate", "pressurehight": "systolic_bp",
                                  "pressurelow": "diastolic_bp", "kcm": "ck_mb"})
        missing = [c for c in meta["features"] if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X_new = raw[meta["features"]].copy()
            X_new = X_new[(X_new["ck_mb"] > 0) & (X_new["troponin"] > 0)]  # needed for log transform
            probs = model.predict_proba(X_new)[:, 1]
            out = raw.loc[X_new.index].copy()
            out["probability_positive"] = probs.round(4)
            out["prediction"] = np.where(probs >= threshold, "positive", "negative")
            st.dataframe(out)
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ==========================================================================
# TAB 2 - MODEL PERFORMANCE
# ==========================================================================
with tab_perf:
    st.header("Model evaluation")
    m = meta["test_metrics_threshold_0.5"]
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{m['accuracy']:.3f}")
    b.metric("Precision", f"{m['precision']:.3f}")
    c.metric("Recall", f"{m['recall_sensitivity']:.3f}")
    d.metric("F1-score", f"{m['f1']:.3f}")
    e.metric("ROC-AUC", f"{meta['test_auc']:.3f}")

    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                      index=["Actual negative", "Actual positive"],
                      columns=["Predicted negative", "Predicted positive"])
    st.subheader("Confusion matrix (test set, threshold 0.50)")
    st.dataframe(cm)

    if FIG_DIR.exists():
        show = [
            ("11_confusion_matrix.png", "Confusion matrix"),
            ("12_roc_curve.png", "ROC curve & AUC"),
            ("13_precision_recall_curve.png", "Precision-Recall curve"),
            ("14_threshold_analysis.png", "Threshold analysis"),
            ("17_coefficients_odds_ratios.png", "Coefficients & odds ratios"),
            ("07_correlation_heatmap.png", "Correlation heatmap"),
        ]
        for fname, caption in show:
            f = FIG_DIR / fname
            if f.exists():
                st.image(str(f), caption=caption)
    else:
        st.info("Run `heart_attack_analysis.py` to generate charts in the `figures/` folder.")

# ==========================================================================
# TAB 3 - ABOUT
# ==========================================================================
with tab_about:
    st.header("About this model")
    st.markdown(f"""
**Algorithm:** Logistic Regression (scikit-learn), tuned with 5-fold cross-validated grid search
(best: `C={meta['best_params']['clf__C']}`, `penalty={meta['best_params']['clf__penalty']}`).

**Pre-processing inside the model pipeline**
- `CK-MB` and `Troponin` are highly right-skewed, so they are log-transformed, then standardised
- `Age`, `Heart rate`, `Systolic BP`, `Diastolic BP`, `Glucose` are standardised
- `Gender` is 1 = male, 0 = female

**Inputs**

| Feature | Meaning |
|---|---|
| Age | years |
| Gender | 1 = male, 0 = female |
| Heart rate | beats per minute |
| Systolic / Diastolic BP | mmHg |
| Glucose | blood glucose, mg/dL |
| CK-MB | creatine kinase-MB enzyme (dataset column `kcm`) |
| Troponin | cardiac troponin (ng/mL) |

**Important limitations**
- Trained on only {1319:,} patients; the deployed model was refit on all of them.
- Predictions are dominated by **troponin** and **CK-MB**, which are themselves diagnostic markers
  of heart-muscle damage - so this model confirms an ongoing event more than it forecasts future risk.
- This is an educational project, **not** a clinical decision tool.
""")
