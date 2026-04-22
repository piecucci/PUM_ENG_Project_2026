"""
MP3: Baseline Modeling & Algorithm Comparison — Full Solution Script
Runs all TODO tasks from mp3_starter.ipynb and prints all results needed for the MCQ test.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
)

# ── 0. Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Load Checkpoint from MP2 ────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[2] / "2. data"
OUTPUT_DIR = Path(__file__).resolve().parent  # save plots next to script

checkpoint_file = CHECKPOINT_DIR / "mp2_checkpoint.pkl"
if not checkpoint_file.exists():
    # Fallback to golden checkpoint
    checkpoint_file = DATA_DIR / "checkpoints" / "checkpoint_for_mp3.pkl"

with open(checkpoint_file, "rb") as f:
    checkpoint = pickle.load(f)

X_train = checkpoint["X_train"]
X_test = checkpoint["X_test"]
y_train = checkpoint["y_train"]
y_test = checkpoint["y_test"]
feature_names = checkpoint["feature_names"]
gender_test = checkpoint.get("gender_test")

print("=" * 70)
print("STEP 1: DATA LOADED")
print("=" * 70)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train lapse rate: {y_train.mean():.4f}")
print(f"y_test  lapse rate: {y_test.mean():.4f}")
print(f"Features ({X_train.shape[1]}): {feature_names}")

# ── 2. Logistic Regression ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: LOGISTIC REGRESSION")
print("=" * 70)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix (LR):")
print(cm_lr)

print("\nClassification Report (LR):")
print(classification_report(y_test, y_pred_lr, digits=4))

auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"ROC-AUC (LR): {auc_lr:.4f}")

# Confusion matrix business interpretation
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
recall_lr_class1 = tp_lr / (tp_lr + fn_lr)
print(f"\nBusiness Interpretation (LR):")
print(f"  TN (correctly identified active): {tn_lr}")
print(f"  FP (active flagged as lapsed — wasted budget): {fp_lr}")
print(f"  FN (lapsed missed — lost opportunity): {fn_lr}")
print(f"  TP (lapsed correctly caught): {tp_lr}")
print(f"  Recall class 1 (lapsed): {recall_lr_class1:.4f}")
print(f"  → Marketing would reach {recall_lr_class1*100:.1f}% of at-risk customers")

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy (LR): {lr_accuracy:.4f}")

# ── 3. Random Forest ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix (RF):")
print(cm_rf)

print("\nClassification Report (RF):")
print(classification_report(y_test, y_pred_rf, digits=4))

auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC-AUC (RF): {auc_rf:.4f}")

tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
recall_rf_class1 = tp_rf / (tp_rf + fn_rf)
print(f"\nBusiness Interpretation (RF):")
print(f"  TN: {tn_rf}, FP: {fp_rf}, FN: {fn_rf}, TP: {tp_rf}")
print(f"  Recall class 1 (lapsed): {recall_rf_class1:.4f}")

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy (RF): {rf_accuracy:.4f}")

# ── 4. ROC Curve Comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: ROC CURVE COMPARISON")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))
RocCurveDisplay.from_predictions(
    y_test, y_prob_lr, name=f"Logistic Regression (AUC={auc_lr:.4f})", ax=ax
)
RocCurveDisplay.from_predictions(
    y_test, y_prob_rf, name=f"Random Forest (AUC={auc_rf:.4f})", ax=ax
)
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC=0.5)")
ax.set_title("ROC Curve Comparison — LR vs RF")
ax.legend()
plt.tight_layout()
roc_path = OUTPUT_DIR / "mp3_roc_comparison.png"
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"ROC curve saved to: {roc_path}")

# Baseline comparison
mp1_auc = 0.83
print(f"\nMP1 Baseline AUC: ~{mp1_auc}")
print(f"MP3 LR AUC:        {auc_lr:.4f}")
print(f"Improvement:        {auc_lr - mp1_auc:.4f}")
print("→ Modest improvement suggests most predictive signal comes from the")
print("  core numeric features, not extra cleaning/encoding from MP2.")

# ── 5. Feature Importance (Random Forest) ──────────────────────────
print("\n" + "=" * 70)
print("STEP 5: FEATURE IMPORTANCE (TOP 15)")
print("=" * 70)

if hasattr(X_train, "columns"):
    feat_names = list(X_train.columns)
else:
    feat_names = feature_names

importances = pd.Series(rf.feature_importances_, index=feat_names)
top15 = importances.nlargest(15)

print("\nTop 15 Feature Importances:")
for i, (name, val) in enumerate(top15.items(), 1):
    print(f"  {i:2d}. {name:35s} {val:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
top15.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
fi_path = OUTPUT_DIR / "mp3_feature_importance.png"
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Feature importance plot saved to: {fi_path}")

print(f"\nMost important feature: {top15.index[0]} ({top15.values[0]:.4f})")

# ── 6. Overfitting Check ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: OVERFITTING CHECK (TRAIN vs TEST)")
print("=" * 70)

lr_train_acc = lr.score(X_train, y_train)
lr_test_acc = lr.score(X_test, y_test)
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

print(f"\nLogistic Regression:")
print(f"  Train accuracy: {lr_train_acc:.4f}")
print(f"  Test accuracy:  {lr_test_acc:.4f}")
print(f"  Gap:            {lr_train_acc - lr_test_acc:.4f}")

print(f"\nRandom Forest:")
print(f"  Train accuracy: {rf_train_acc:.4f}")
print(f"  Test accuracy:  {rf_test_acc:.4f}")
print(f"  Gap:            {rf_train_acc - rf_test_acc:.4f}")

print("\nInterpretation:")
print("  RF achieves ~100% train accuracy but lower test accuracy → overfitting.")
print("  LR achieves similar accuracy on both → generalizes well, no overfitting.")
print("  RF memorizes training data (deep trees). LR is more trustworthy here.")
print("  However, RF still has higher AUC, suggesting it captures useful patterns.")

# ── 7. Model Comparison Summary Table ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: MODEL COMPARISON SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (class 1)", "Recall (class 1)",
               "F1-Score (class 1)", "ROC-AUC", "Train Accuracy"],
    "Logistic Regression": [
        round(accuracy_score(y_test, y_pred_lr), 4),
        round(precision_score(y_test, y_pred_lr), 4),
        round(recall_score(y_test, y_pred_lr), 4),
        round(f1_score(y_test, y_pred_lr), 4),
        round(auc_lr, 4),
        round(lr_train_acc, 4),
    ],
    "Random Forest": [
        round(accuracy_score(y_test, y_pred_rf), 4),
        round(precision_score(y_test, y_pred_rf), 4),
        round(recall_score(y_test, y_pred_rf), 4),
        round(f1_score(y_test, y_pred_rf), 4),
        round(auc_rf, 4),
        round(rf_train_acc, 4),
    ],
})

print("\n" + summary.to_string(index=False))

print("\n\nBusiness Recommendation:")
print("  For MajsterPlus reactivation campaign, Logistic Regression may be preferred:")
print("  - Similar test accuracy → reliable predictions")
print("  - No overfitting → trustworthy on new data")
print("  - Interpretable → can explain to stakeholders")
print("  - RF has slightly better AUC but overfits significantly")
print("  - In MP4, cost analysis may change this recommendation")

# ── 8. Save Checkpoint for MP4 ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: SAVING CHECKPOINT FOR MP4")
print("=" * 70)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_mp3 = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": list(X_train.columns) if hasattr(X_train, "columns") else feature_names,
    "gender_test": gender_test,
    "lr_model": lr,
    "rf_model": rf,
    "y_pred_lr": y_pred_lr,
    "y_prob_lr": y_prob_lr,
    "y_pred_rf": y_pred_rf,
    "y_prob_rf": y_prob_rf,
}

mp3_ckpt_path = CHECKPOINT_DIR / "mp3_checkpoint.pkl"
with open(mp3_ckpt_path, "wb") as f:
    pickle.dump(checkpoint_mp3, f)
print(f"Checkpoint saved: {mp3_ckpt_path}")

# ── SUMMARY OF KEY VALUES FOR MCQ TEST ────────────────────────────
print("\n" + "=" * 70)
print("🎯 KEY VALUES FOR MCQ TEST")
print("=" * 70)
print(f"  LR Test Accuracy:           {lr_test_acc:.4f}")
print(f"  LR Train Accuracy:          {lr_train_acc:.4f}")
print(f"  LR ROC-AUC:                 {auc_lr:.4f}")
print(f"  LR Recall (class 1):        {recall_lr_class1:.4f}")
print(f"  LR Precision (class 1):     {precision_score(y_test, y_pred_lr):.4f}")
print(f"  LR F1 (class 1):            {f1_score(y_test, y_pred_lr):.4f}")
print()
print(f"  RF Test Accuracy:           {rf_test_acc:.4f}")
print(f"  RF Train Accuracy:          {rf_train_acc:.4f}")
print(f"  RF ROC-AUC:                 {auc_rf:.4f}")
print(f"  RF Recall (class 1):        {recall_rf_class1:.4f}")
print(f"  RF Precision (class 1):     {precision_score(y_test, y_pred_rf):.4f}")
print(f"  RF F1 (class 1):            {f1_score(y_test, y_pred_rf):.4f}")
print()
print(f"  Most Important Feature:     {top15.index[0]}")
print(f"  2nd Most Important:         {top15.index[1]}")
print(f"  3rd Most Important:         {top15.index[2]}")
print()
print(f"  LR Confusion Matrix:        TN={tn_lr}, FP={fp_lr}, FN={fn_lr}, TP={tp_lr}")
print(f"  RF Confusion Matrix:        TN={tn_rf}, FP={fp_rf}, FN={fn_rf}, TP={tp_rf}")
print()
print(f"  Overfitting: LR gap = {lr_train_acc - lr_test_acc:.4f}, RF gap = {rf_train_acc - rf_test_acc:.4f}")
print(f"  MP1 baseline AUC ≈ 0.83, MP3 LR AUC = {auc_lr:.4f} (diff = {auc_lr - 0.83:.4f})")
