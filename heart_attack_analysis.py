"""
Heart Attack Prediction - Logistic Regression
=============================================
End-to-end pipeline:
  1. Load & inspect data
  2. Preprocessing (cleaning, outlier handling, encoding)
  3. Exploratory Data Analysis (EDA) with charts
  4. Train/test split + sklearn Pipeline (log-transform + scaling + LogisticRegression)
  5. Hyper-parameter tuning (GridSearchCV) + cross-validation
  6. Evaluation: Confusion Matrix, Precision, Recall, F1, ROC, AUC, PR curve
  7. Model interpretation (coefficients / odds ratios)
  8. Save model for Streamlit deployment (heart_attack_model.joblib)

Run:  python heart_attack_analysis.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Heart_Attack.csv"
FIG_DIR = BASE_DIR / "figures"
MODEL_PATH = BASE_DIR / "heart_attack_model.joblib"
RANDOM_STATE = 42
FIG_DIR.mkdir(exist_ok=True)

NEG_COLOR, POS_COLOR = "#2E86AB", "#E4572E"   # negative = blue, positive = red
PALETTE = {"negative": NEG_COLOR, "positive": POS_COLOR}
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.titleweight": "bold"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


# ==========================================================================
# 1. LOAD DATA
# ==========================================================================
print("=" * 70, "\n1. LOAD DATA\n" + "=" * 70)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum().to_string())
print("\nDuplicate rows:", df.duplicated().sum())

# Friendlier column names (original dataset has typos such as 'impluse')
df = df.rename(columns={
    "impluse": "heart_rate",
    "pressurehight": "systolic_bp",
    "pressurelow": "diastolic_bp",
    "kcm": "ck_mb",          # CK-MB enzyme (creatine kinase - MB)
})

# ==========================================================================
# 2. PREPROCESSING
# ==========================================================================
print("\n" + "=" * 70, "\n2. PREPROCESSING\n" + "=" * 70)

# 2.1 Data-entry error: heart_rate = 1111 is physiologically impossible
#     (next highest value is 135) -> almost certainly a typo for 111.
bad = df["heart_rate"] > 300
print(f"Impossible heart-rate rows fixed (1111 -> 111): {int(bad.sum())}")
df.loc[bad, "heart_rate"] = 111

# 2.2 Encode target: positive = 1 (heart attack), negative = 0
df["target"] = (df["class"] == "positive").astype(int)

FEATURES = ["age", "gender", "heart_rate", "systolic_bp", "diastolic_bp",
            "glucose", "ck_mb", "troponin"]
LOG_FEATURES = ["ck_mb", "troponin"]                     # extremely right-skewed
SCALE_ONLY = ["age", "heart_rate", "systolic_bp", "diastolic_bp", "glucose"]
PASSTHROUGH = ["gender"]                                 # already 0/1 (1 = male)

print("\nSkewness BEFORE transform:")
print(df[FEATURES].skew().round(2).to_string())
print("\nSkewness of ck_mb / troponin AFTER log transform:")
print(np.log(df[LOG_FEATURES]).skew().round(2).to_string())
print("\nNOTE: extreme ck_mb / troponin values are NOT removed - they are the")
print("clinical signal of a heart attack. Log-transform tames them instead.")

# ==========================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================================================
print("\n" + "=" * 70, "\n3. EDA\n" + "=" * 70)
print(df[FEATURES].describe().T.round(3).to_string())

# 3.1 Class distribution ---------------------------------------------------
counts = df["class"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.barplot(x=counts.index, y=counts.values, palette=PALETTE, ax=ax[0])
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 10, str(v), ha="center", fontweight="bold")
ax[0].set_title("Class Distribution (counts)")
ax[0].set_xlabel("")
ax[0].set_ylabel("Patients")
ax[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
          colors=[PALETTE[c] for c in counts.index], wedgeprops={"edgecolor": "white"})
ax[1].set_title("Class Distribution (%)")
save(fig, "01_class_distribution.png")

# 3.2 Raw feature distributions -------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, col in zip(axes.ravel(), FEATURES):
    if col == "gender":
        sns.countplot(x=df[col].map({0: "Female", 1: "Male"}), color="#7A9E9F", ax=ax)
    else:
        sns.histplot(df[col], kde=True, color="#5B7DB1", ax=ax)
    ax.set_title(f"{col}  (skew={df[col].skew():.2f})" if col != "gender" else "gender")
    ax.set_xlabel("")
fig.suptitle("Feature Distributions (after fixing the 1111 heart-rate error)",
             fontsize=14, fontweight="bold", y=1.02)
save(fig, "02_feature_distributions.png")

# 3.3 Log-transform effect ---------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for j, col in enumerate(LOG_FEATURES):
    sns.histplot(df[col], kde=True, color="#B5533C", ax=axes[0, j])
    axes[0, j].set_title(f"{col} - raw (skew={df[col].skew():.2f})")
    sns.histplot(np.log(df[col]), kde=True, color="#3C8D5A", ax=axes[1, j])
    axes[1, j].set_title(f"log({col}) (skew={np.log(df[col]).skew():.2f})")
    axes[0, j].set_xlabel("")
    axes[1, j].set_xlabel("")
fig.suptitle("Why we log-transform ck_mb and troponin", fontsize=14, fontweight="bold")
save(fig, "03_log_transform_effect.png")

# 3.4 Boxplots (outlier view) by class ----------------------------------------
num_cols = [c for c in FEATURES if c != "gender"]
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(data=df, x="class", y=col, order=["negative", "positive"],
                palette=PALETTE, ax=ax, fliersize=2.5)
    ax.set_title(col)
    ax.set_xlabel("")
    if col in LOG_FEATURES:
        ax.set_yscale("log")
        ax.set_title(f"{col} (log scale)")
axes.ravel()[-1].axis("off")
fig.suptitle("Feature Distributions by Class (boxplots)", fontsize=14, fontweight="bold", y=1.02)
save(fig, "04_boxplots_by_class.png")

# 3.5 Violin plots for the strongest biomarkers ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (col, data) in zip(axes, [("troponin", np.log(df["troponin"])),
                                  ("ck_mb", np.log(df["ck_mb"])),
                                  ("age", df["age"])]):
    tmp = pd.DataFrame({"class": df["class"], col: data})
    sns.violinplot(data=tmp, x="class", y=col, order=["negative", "positive"],
                   palette=PALETTE, inner="quartile", ax=ax)
    ax.set_title(f"{'log ' if col != 'age' else ''}{col} by class")
    ax.set_xlabel("")
save(fig, "05_violin_key_features.png")

# 3.6 Categorical: gender & age group vs. outcome -------------------------------
df["gender_label"] = df["gender"].map({0: "Female", 1: "Male"})
df["age_group"] = pd.cut(df["age"], [0, 40, 50, 60, 70, 120],
                         labels=["<40", "40-49", "50-59", "60-69", "70+"])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
sns.countplot(data=df, x="gender_label", hue="class", palette=PALETTE, ax=axes[0])
axes[0].set_title("Gender vs Outcome (counts)")
axes[0].set_xlabel("")
rate_g = df.groupby("gender_label")["target"].mean() * 100
sns.barplot(x=rate_g.index, y=rate_g.values, color="#7A9E9F", ax=axes[1])
for i, v in enumerate(rate_g.values):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
axes[1].set_title("Heart-attack rate by Gender (%)")
axes[1].set_ylim(0, 100)
axes[1].set_xlabel("")
rate_a = df.groupby("age_group", observed=True)["target"].mean() * 100
sns.barplot(x=rate_a.index.astype(str), y=rate_a.values, color="#C9803B", ax=axes[2])
for i, v in enumerate(rate_a.values):
    axes[2].text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
axes[2].set_title("Heart-attack rate by Age group (%)")
axes[2].set_ylim(0, 100)
axes[2].set_xlabel("")
save(fig, "06_gender_age_vs_outcome.png")

# 3.7 Correlation heatmap ---------------------------------------------------------
corr_df = df[FEATURES + ["target"]].copy()
corr_df["ck_mb"] = np.log(corr_df["ck_mb"])
corr_df["troponin"] = np.log(corr_df["troponin"])
corr_df = corr_df.rename(columns={"ck_mb": "log_ck_mb", "troponin": "log_troponin"})
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix (with log-transformed biomarkers)")
save(fig, "07_correlation_heatmap.png")

print("\nCorrelation with target (sorted):")
print(corr["target"].drop("target").sort_values(ascending=False).round(3).to_string())

# 3.8 Multicollinearity (VIF) -------------------------------------------------------
vif_X = corr_df.drop(columns="target")
inv_corr = np.linalg.inv(vif_X.corr().values)
vif = pd.Series(np.diag(inv_corr), index=vif_X.columns).sort_values(ascending=False)
print("\nVariance Inflation Factors (VIF < 5 is fine):")
print(vif.round(2).to_string())
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=vif.values, y=vif.index, color="#6C8EBF", ax=ax)
ax.axvline(5, color="red", ls="--", label="VIF = 5 threshold")
ax.set_title("Multicollinearity check (VIF)")
ax.legend()
save(fig, "08_vif.png")

# 3.9 Pairplot of most informative features ------------------------------------------
pp = pd.DataFrame({
    "age": df["age"], "glucose": df["glucose"],
    "log_ck_mb": np.log(df["ck_mb"]), "log_troponin": np.log(df["troponin"]),
    "class": df["class"],
})
g = sns.pairplot(pp, hue="class", palette=PALETTE, corner=True,
                 plot_kws={"alpha": 0.5, "s": 18}, height=2.2)
g.figure.suptitle("Pairplot - key features", y=1.02, fontsize=14, fontweight="bold")
g.figure.savefig(FIG_DIR / "09_pairplot.png", bbox_inches="tight")
plt.close(g.figure)
print("  saved figures/09_pairplot.png")

# ==========================================================================
# 4. TRAIN / TEST SPLIT + PIPELINE
# ==========================================================================
print("\n" + "=" * 70, "\n4. TRAIN / TEST SPLIT & PIPELINE\n" + "=" * 70)
X = df[FEATURES]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print("Train class share:", y_train.mean().round(3), "| Test class share:", y_test.mean().round(3))

# All preprocessing lives INSIDE the pipeline -> no data leakage, and the same
# steps are applied automatically at prediction time in the Streamlit app.
preprocess = ColumnTransformer([
    ("log_scale", Pipeline([("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                            ("scale", StandardScaler())]), LOG_FEATURES),
    ("scale", StandardScaler(), SCALE_ONLY),
    ("pass", "passthrough", PASSTHROUGH),
])
pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
])

# ==========================================================================
# 5. HYPER-PARAMETER TUNING + CROSS-VALIDATION
# ==========================================================================
print("\n" + "=" * 70, "\n5. TUNING (GridSearchCV, 5-fold stratified, scoring=ROC-AUC)\n" + "=" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grid = {
    "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__class_weight": [None, "balanced"],
}
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best params :", grid.best_params_)
print(f"Best CV AUC : {grid.best_score_:.4f}")

# Baseline (defaults) for comparison
base = pipe.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])
print(f"Baseline (default LR) test AUC: {base_auc:.4f}")

# Regularisation curve
res = pd.DataFrame(grid.cv_results_)
fig, ax = plt.subplots(figsize=(8, 4.5))
for (pen, cw), sub in res.groupby([res["param_clf__penalty"], res["param_clf__class_weight"].astype(str)]):
    sub = sub.sort_values("param_clf__C")
    ax.plot(sub["param_clf__C"].astype(float), sub["mean_test_score"], marker="o",
            label=f"{pen}, class_weight={cw}")
ax.set_xscale("log")
ax.set_xlabel("C (inverse regularisation strength)")
ax.set_ylabel("Mean CV ROC-AUC")
ax.set_title("GridSearchCV results")
ax.legend(fontsize=8)
save(fig, "10_gridsearch_results.png")

# 5-fold CV of the tuned model on the training set (stability check)
cv_res = cross_validate(model, X_train, y_train, cv=cv,
                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_summary = {k.replace("test_", ""): (v.mean(), v.std()) for k, v in cv_res.items() if k.startswith("test_")}
print("\n5-fold CV on training data (mean +/- std):")
for k, (m, s) in cv_summary.items():
    print(f"  {k:10s}: {m:.4f} +/- {s:.4f}")

# ==========================================================================
# 6. EVALUATION ON HELD-OUT TEST SET
# ==========================================================================
print("\n" + "=" * 70, "\n6. MODEL EVALUATION (test set)\n" + "=" * 70)
proba_test = model.predict_proba(X_test)[:, 1]

# Out-of-fold probabilities on TRAIN -> used to pick a threshold without touching the test set
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thr_grid = np.linspace(0.05, 0.95, 91)
f1_oof = [f1_score(y_train, oof >= t) for t in thr_grid]
best_thr = float(thr_grid[int(np.argmax(f1_oof))])
print(f"F1-optimal threshold (from CV on train): {best_thr:.2f}")


def metrics_at(thr):
    pred = (proba_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "threshold": round(float(thr), 3),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall_sensitivity": recall_score(y_test, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, pred),
        "mcc": matthews_corrcoef(y_test, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


m_default, m_tuned = metrics_at(0.5), metrics_at(best_thr)
auc = roc_auc_score(y_test, proba_test)
ap = average_precision_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)

print(f"\nROC-AUC: {auc:.4f} | PR-AUC (avg precision): {ap:.4f} | Brier: {brier:.4f}")
print("\n--- Threshold = 0.50 ---")
print(pd.Series(m_default).round(4).to_string())
print("\n--- Classification report @0.50 ---")
print(classification_report(y_test, (proba_test >= 0.5).astype(int),
                            target_names=["negative", "positive"], digits=4))
print(f"--- Threshold = {best_thr:.2f} (F1-optimal via CV) ---")
print(pd.Series(m_tuned).round(4).to_string())

# 6.1 Confusion matrix (counts + normalised) ---------------------------------------------
pred50 = (proba_test >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred50)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
axes[0].set_title("Confusion Matrix (counts) - threshold 0.50")
ConfusionMatrixDisplay(confusion_matrix(y_test, pred50, normalize="true"),
                       display_labels=["negative", "positive"]).plot(
    ax=axes[1], cmap="Greens", colorbar=False, values_format=".2%")
axes[1].set_title("Confusion Matrix (row-normalised)")
for a in axes:
    a.grid(False)
save(fig, "11_confusion_matrix.png")

# 6.2 ROC curve ---------------------------------------------------------------------------------
fpr, tpr, roc_thr = roc_curve(y_test, proba_test)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr, tpr, color=POS_COLOR, lw=2.5, label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.5)")
ax.fill_between(fpr, tpr, alpha=0.12, color=POS_COLOR)
op = np.argmin(np.abs(roc_thr - 0.5))
ax.scatter(fpr[op], tpr[op], s=80, color="black", zorder=5, label="Operating point @0.50")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
ax.set_title("ROC Curve - test set")
ax.legend(loc="lower right")
save(fig, "12_roc_curve.png")

# 6.3 Precision-Recall curve -----------------------------------------------------------------------
prec, rec, _ = precision_recall_curve(y_test, proba_test)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(rec, prec, color="#6A4C93", lw=2.5, label=f"PR curve (AP = {ap:.3f})")
ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline = {y_test.mean():.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - test set")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
save(fig, "13_precision_recall_curve.png")

# 6.4 Threshold analysis (from CV on train) ---------------------------------------------------------
p_t = [precision_score(y_train, oof >= t) for t in thr_grid]
r_t = [recall_score(y_train, oof >= t) for t in thr_grid]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(thr_grid, p_t, label="Precision", lw=2)
ax.plot(thr_grid, r_t, label="Recall", lw=2)
ax.plot(thr_grid, f1_oof, label="F1", lw=2.5, color="black")
ax.axvline(0.5, color="grey", ls=":", label="Default 0.50")
ax.axvline(best_thr, color="red", ls="--", label=f"F1-optimal {best_thr:.2f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs. Threshold (out-of-fold on training data)")
ax.legend()
save(fig, "14_threshold_analysis.png")

# 6.5 Predicted-probability distribution ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
sns.histplot(x=proba_test, hue=y_test.map({0: "negative", 1: "positive"}), bins=30,
             palette=PALETTE, stat="count", multiple="layer", alpha=0.6, ax=ax)
ax.axvline(0.5, color="black", ls="--")
ax.set_xlabel("Predicted probability of heart attack")
ax.set_title("Model confidence: predicted probability by true class")
save(fig, "15_probability_distribution.png")

# 6.6 Metrics summary bar -------------------------------------------------------------------------------
keys = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1"]
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(keys))
w = 0.38
b1 = ax.bar(x - w / 2, [m_default[k] for k in keys], w, label="Threshold 0.50", color="#2E86AB")
b2 = ax.bar(x + w / 2, [m_tuned[k] for k in keys], w, label=f"Threshold {best_thr:.2f}", color="#F6AE2D")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.3f}",
                ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "Specificity", "F1"])
ax.set_ylim(0, 1.12)
ax.set_title("Test-set metrics")
ax.legend()
save(fig, "16_metrics_summary.png")

# ==========================================================================
# 7. INTERPRETATION - COEFFICIENTS & ODDS RATIOS
# ==========================================================================
print("\n" + "=" * 70, "\n7. MODEL INTERPRETATION\n" + "=" * 70)
feat_names = ["log(ck_mb)", "log(troponin)"] + SCALE_ONLY + PASSTHROUGH
coefs = model.named_steps["clf"].coef_.ravel()
coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs,
                        "odds_ratio": np.exp(coefs)}).sort_values("coef", key=abs, ascending=False)
print("(features are standardised: odds_ratio = change in odds per +1 SD; gender is 0/1)")
print(coef_df.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
order = coef_df.sort_values("coef")
axes[0].barh(order["feature"], order["coef"],
             color=[POS_COLOR if c > 0 else NEG_COLOR for c in order["coef"]])
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("Logistic Regression Coefficients\n(red = raises risk, blue = lowers risk)")
axes[1].barh(order["feature"], order["odds_ratio"] - 1, left=1,
             color=[POS_COLOR if c > 1 else NEG_COLOR for c in order["odds_ratio"]])
axes[1].set_xscale("log")          # troponin's OR is ~290x, log scale keeps others visible
axes[1].axvline(1, color="black", lw=1)
axes[1].set_title("Odds Ratios per +1 SD (log scale)")
save(fig, "17_coefficients_odds_ratios.png")

# ==========================================================================
# 8. SAVE MODEL + METADATA FOR STREAMLIT
# ==========================================================================
print("\n" + "=" * 70, "\n8. SAVE MODEL\n" + "=" * 70)
final_model = grid.best_estimator_
# Refit on ALL data for deployment (hyper-parameters already chosen on train CV).
final_model.fit(X, y)
joblib.dump(final_model, MODEL_PATH)

# Ranges shown in the Streamlit app (from the cleaned data)
ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
              "median": float(df[c].median())} for c in FEATURES}
meta = {
    "features": FEATURES,
    "best_params": {k: (v if v is None or isinstance(v, str) else float(v))
                    for k, v in grid.best_params_.items()},
    "cv_auc": float(grid.best_score_),
    "test_auc": float(auc),
    "test_metrics_threshold_0.5": m_default,
    "test_metrics_tuned_threshold": m_tuned,
    "f1_optimal_threshold": best_thr,
    "ranges": ranges,
}
(BASE_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Model saved  -> {MODEL_PATH.name}")
print("Metadata     -> model_metadata.json")
print("Figures      -> figures/")
print("\nDONE.")
