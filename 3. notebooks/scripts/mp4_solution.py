"""
MP4: Model Evaluation & Business Impact — Full Solution Script
Runs all TODO tasks from mp4_starter.ipynb and prints all results needed for the MCQ test.
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
from sklearn.metrics import confusion_matrix

# ── 0. Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Load Checkpoint from MP3 ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "2. data"
OUTPUT_DIR = Path(__file__).resolve().parent

checkpoint_file = CHECKPOINT_DIR / "mp3_checkpoint.pkl"
if not checkpoint_file.exists():
    checkpoint_file = DATA_DIR / "checkpoints" / "checkpoint_for_mp4.pkl"

with open(checkpoint_file, "rb") as f:
    checkpoint = pickle.load(f)

X_train = checkpoint["X_train"]
X_test = checkpoint["X_test"]
y_train = checkpoint["y_train"]
y_test = checkpoint["y_test"]
feature_names = checkpoint["feature_names"]
gender_test = checkpoint.get("gender_test")
lr_model = checkpoint.get("lr_model")
rf_model = checkpoint.get("rf_model")
y_prob_lr = checkpoint.get("y_prob_lr")
y_prob_rf = checkpoint.get("y_prob_rf")

print("=" * 70)
print("STEP 1: DATA LOADED FROM MP3 CHECKPOINT")
print("=" * 70)
print(f"Loaded keys: {list(checkpoint.keys())}")
print(f"Test set size: {len(y_test)}")
print(f"y_test lapse rate: {y_test.mean():.4f}")

# ── 2. Business Parameters ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: BUSINESS PARAMETERS")
print("=" * 70)

CAMPAIGN_COST = 80  # PLN per customer contacted

# Load raw data to get original total_spend values (checkpoint has scaled data)
cust_raw = pd.read_csv(DATA_DIR / "customers.csv")
cust_raw["total_spend_numeric"] = (
    cust_raw["total_spend"]
    .str.replace("PLN ", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Get median total_spend of lapsed customers in test set
lapsed_test_idx = y_test[y_test == 1].index
lapsed_test_spend = cust_raw.loc[lapsed_test_idx, "total_spend_numeric"]
EXPECTED_REVENUE = lapsed_test_spend.median()

print(f"Campaign cost per customer: {CAMPAIGN_COST} PLN")
print(f"Expected revenue if reactivated (median lapsed test spend): {EXPECTED_REVENUE:.2f} PLN")
print(f"Net gain per TP: {EXPECTED_REVENUE - CAMPAIGN_COST:.2f} PLN")
print(f"Loss per FP: {CAMPAIGN_COST} PLN")
print(f"Number of lapsed test customers: {len(lapsed_test_spend)}")

# ── 3. Cost Matrix / compute_profit function ───────────────────────
print("\n" + "=" * 70)
print("STEP 3: COST MATRIX CONSTRUCTION")
print("=" * 70)


def compute_profit(y_true, y_pred, revenue, cost):
    """
    Compute total campaign profit given true labels and predictions.
    
    - TP (lapsed, contacted): revenue - cost
    - FP (active, contacted): -cost
    - FN (lapsed, not contacted): 0
    - TN (active, not contacted): 0
    
    Returns: (total_profit, tp, fp, fn, tn)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    total_profit = tp * (revenue - cost) + fp * (-cost)
    
    return total_profit, tp, fp, fn, tn


print("compute_profit() function defined.")
print(f"Cost matrix:")
print(f"  TP gain: {EXPECTED_REVENUE} - {CAMPAIGN_COST} = {EXPECTED_REVENUE - CAMPAIGN_COST:.2f} PLN")
print(f"  FP loss: -{CAMPAIGN_COST} PLN")
print(f"  FN:      0 PLN")
print(f"  TN:      0 PLN")

# Think About This: asymmetry
print("\nAsymmetry analysis:")
print(f"  TP net gain: {EXPECTED_REVENUE - CAMPAIGN_COST:.2f} PLN")
print(f"  FP net loss: {CAMPAIGN_COST} PLN")
print(f"  Since FP loss ({CAMPAIGN_COST}) > TP gain ({EXPECTED_REVENUE - CAMPAIGN_COST:.0f}),")
print(f"  the model should be CONSERVATIVE — predict fewer, more confident positives.")
print(f"  Each false positive costs more than each true positive earns.")

# ── 4. Profit at Threshold = 0.5 ──────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: PROFIT AT THRESHOLD = 0.5")
print("=" * 70)

y_pred_lr_05 = (y_prob_lr >= 0.5).astype(int)
y_pred_rf_05 = (y_prob_rf >= 0.5).astype(int)

profit_lr_05, tp_lr, fp_lr, fn_lr, tn_lr = compute_profit(
    y_test, y_pred_lr_05, EXPECTED_REVENUE, CAMPAIGN_COST
)
profit_rf_05, tp_rf, fp_rf, fn_rf, tn_rf = compute_profit(
    y_test, y_pred_rf_05, EXPECTED_REVENUE, CAMPAIGN_COST
)

n_test = len(y_test)

print(f"\nLogistic Regression (threshold=0.5):")
print(f"  TP={tp_lr}, FP={fp_lr}, FN={fn_lr}, TN={tn_lr}")
print(f"  Contacted: {tp_lr + fp_lr}")
print(f"  Total profit: {profit_lr_05:.2f} PLN")
print(f"  Profit per record: {profit_lr_05 / n_test:.2f} PLN")

print(f"\nRandom Forest (threshold=0.5):")
print(f"  TP={tp_rf}, FP={fp_rf}, FN={fn_rf}, TN={tn_rf}")
print(f"  Contacted: {tp_rf + fp_rf}")
print(f"  Total profit: {profit_rf_05:.2f} PLN")
print(f"  Profit per record: {profit_rf_05 / n_test:.2f} PLN")

# ── 5. Baseline Comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: BASELINE COMPARISON")
print("=" * 70)

# Contact everyone
y_all_ones = np.ones(n_test, dtype=int)
profit_all, tp_all, fp_all, fn_all, tn_all = compute_profit(
    y_test, y_all_ones, EXPECTED_REVENUE, CAMPAIGN_COST
)

# Contact nobody
y_all_zeros = np.zeros(n_test, dtype=int)
profit_none, tp_none, fp_none, fn_none, tn_none = compute_profit(
    y_test, y_all_zeros, EXPECTED_REVENUE, CAMPAIGN_COST
)

print(f"\nContact Everyone (all predict lapsed):")
print(f"  TP={tp_all}, FP={fp_all}, FN={fn_all}, TN={tn_all}")
print(f"  Total profit: {profit_all:.2f} PLN")
print(f"  Profit per record: {profit_all / n_test:.2f} PLN")

print(f"\nContact Nobody (all predict active):")
print(f"  TP={tp_none}, FP={fp_none}, FN={fn_none}, TN={tn_none}")
print(f"  Total profit: {profit_none:.2f} PLN")
print(f"  Profit per record: {profit_none / n_test:.2f} PLN")

# Explain why contact everyone loses money
print(f"\nWhy 'Contact Everyone' loses money:")
print(f"  FP count: {fp_all} (active customers contacted)")
print(f"  FP loss:  {fp_all} × {CAMPAIGN_COST} = {fp_all * CAMPAIGN_COST:.2f} PLN")
print(f"  TP count: {tp_all} (lapsed customers caught)")
print(f"  TP gain:  {tp_all} × ({EXPECTED_REVENUE} - {CAMPAIGN_COST}) = {tp_all * (EXPECTED_REVENUE - CAMPAIGN_COST):.2f} PLN")
print(f"  Net:      {tp_all * (EXPECTED_REVENUE - CAMPAIGN_COST):.2f} - {fp_all * CAMPAIGN_COST:.2f} = {profit_all:.2f} PLN")
print(f"  80% of customers are active → massive FP loss overwhelms TP gain.")

# ── 6. Threshold Optimization ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: THRESHOLD OPTIMIZATION")
print("=" * 70)

thresholds = np.arange(0.05, 1.0, 0.05)
profits_lr = []
profits_rf = []
details_lr = []
details_rf = []

for th in thresholds:
    y_pred_lr_th = (y_prob_lr >= th).astype(int)
    y_pred_rf_th = (y_prob_rf >= th).astype(int)
    
    p_lr, tp_l, fp_l, fn_l, tn_l = compute_profit(y_test, y_pred_lr_th, EXPECTED_REVENUE, CAMPAIGN_COST)
    p_rf, tp_r, fp_r, fn_r, tn_r = compute_profit(y_test, y_pred_rf_th, EXPECTED_REVENUE, CAMPAIGN_COST)
    
    profits_lr.append(p_lr)
    profits_rf.append(p_rf)
    details_lr.append((th, p_lr, tp_l, fp_l, fn_l, tn_l))
    details_rf.append((th, p_rf, tp_r, fp_r, fn_r, tn_r))

# Print threshold sweep results
print("\nThreshold sweep results:")
print(f"{'Threshold':>10} | {'LR Profit':>12} | {'LR TP':>6} | {'LR FP':>6} | {'RF Profit':>12} | {'RF TP':>6} | {'RF FP':>6}")
print("-" * 80)
for i, th in enumerate(thresholds):
    lr_d = details_lr[i]
    rf_d = details_rf[i]
    print(f"{th:10.2f} | {lr_d[1]:12.2f} | {lr_d[2]:6d} | {lr_d[3]:6d} | {rf_d[1]:12.2f} | {rf_d[2]:6d} | {rf_d[3]:6d}")

# Plot profit vs threshold
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, profits_lr, "b-o", label="Logistic Regression", markersize=5)
ax.plot(thresholds, profits_rf, "r-s", label="Random Forest", markersize=5)
ax.axhline(y=profit_all, color="gray", linestyle="--", alpha=0.7, label=f"Contact Everyone ({profit_all:.0f} PLN)")
ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Total Profit (PLN)")
ax.set_title("Profit vs. Classification Threshold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mp4_profit_vs_threshold.png", dpi=150)
plt.close()
print(f"\nProfit vs threshold plot saved.")

# ── 7. Optimal Threshold Analysis ─────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: OPTIMAL THRESHOLD ANALYSIS")
print("=" * 70)

# Find optimal thresholds
best_lr_idx = np.argmax(profits_lr)
best_rf_idx = np.argmax(profits_rf)

optimal_th_lr = thresholds[best_lr_idx]
optimal_profit_lr = profits_lr[best_lr_idx]
optimal_lr_details = details_lr[best_lr_idx]

optimal_th_rf = thresholds[best_rf_idx]
optimal_profit_rf = profits_rf[best_rf_idx]
optimal_rf_details = details_rf[best_rf_idx]

print(f"\nLogistic Regression — Optimal Threshold:")
print(f"  Threshold: {optimal_th_lr:.2f}")
print(f"  Max profit: {optimal_profit_lr:.2f} PLN")
print(f"  TP={optimal_lr_details[2]}, FP={optimal_lr_details[3]}, FN={optimal_lr_details[4]}, TN={optimal_lr_details[5]}")
print(f"  Customers contacted: {optimal_lr_details[2] + optimal_lr_details[3]}")
print(f"  Profit per record: {optimal_profit_lr / n_test:.2f} PLN")

print(f"\nRandom Forest — Optimal Threshold:")
print(f"  Threshold: {optimal_th_rf:.2f}")
print(f"  Max profit: {optimal_profit_rf:.2f} PLN")
print(f"  TP={optimal_rf_details[2]}, FP={optimal_rf_details[3]}, FN={optimal_rf_details[4]}, TN={optimal_rf_details[5]}")
print(f"  Customers contacted: {optimal_rf_details[2] + optimal_rf_details[3]}")
print(f"  Profit per record: {optimal_profit_rf / n_test:.2f} PLN")

# Breakdown at optimal threshold (for RF)
print(f"\nProfit breakdown at optimal RF threshold ({optimal_th_rf:.2f}):")
tp_opt = optimal_rf_details[2]
fp_opt = optimal_rf_details[3]
tp_gain = tp_opt * (EXPECTED_REVENUE - CAMPAIGN_COST)
fp_loss = fp_opt * CAMPAIGN_COST
print(f"  (a) TP gain: {tp_opt} × ({EXPECTED_REVENUE:.0f} - {CAMPAIGN_COST}) = {tp_gain:.2f} PLN")
print(f"  (b) FP loss: {fp_opt} × {CAMPAIGN_COST} = {fp_loss:.2f} PLN")
print(f"  (c) Net profit: {tp_gain:.2f} - {fp_loss:.2f} = {tp_gain - fp_loss:.2f} PLN")

# Also for LR
print(f"\nProfit breakdown at optimal LR threshold ({optimal_th_lr:.2f}):")
tp_opt_lr = optimal_lr_details[2]
fp_opt_lr = optimal_lr_details[3]
tp_gain_lr = tp_opt_lr * (EXPECTED_REVENUE - CAMPAIGN_COST)
fp_loss_lr = fp_opt_lr * CAMPAIGN_COST
print(f"  (a) TP gain: {tp_opt_lr} × ({EXPECTED_REVENUE:.0f} - {CAMPAIGN_COST}) = {tp_gain_lr:.2f} PLN")
print(f"  (b) FP loss: {fp_opt_lr} × {CAMPAIGN_COST} = {fp_loss_lr:.2f} PLN")
print(f"  (c) Net profit: {tp_gain_lr:.2f} - {fp_loss_lr:.2f} = {tp_gain_lr - fp_loss_lr:.2f} PLN")

# What if cost increased by 10 PLN?
print(f"\nSensitivity: What if campaign cost = 90 PLN?")
for model_name, th, tp_v, fp_v in [
    ("LR", optimal_th_lr, optimal_lr_details[2], optimal_lr_details[3]),
    ("RF", optimal_th_rf, optimal_rf_details[2], optimal_rf_details[3]),
]:
    new_profit = tp_v * (EXPECTED_REVENUE - 90) + fp_v * (-90)
    print(f"  {model_name} at th={th:.2f}: profit = {new_profit:.2f} PLN (was {tp_v * (EXPECTED_REVENUE - CAMPAIGN_COST) - fp_v * CAMPAIGN_COST:.2f} PLN)")

# Why threshold below 0.5 can be optimal
print(f"\nWhy optimal threshold differs from 0.5:")
print(f"  At th=0.5: LR profit = {profit_lr_05:.2f} PLN, RF profit = {profit_rf_05:.2f} PLN")
print(f"  At optimal: LR profit = {optimal_profit_lr:.2f} PLN (th={optimal_th_lr:.2f})")
print(f"                RF profit = {optimal_profit_rf:.2f} PLN (th={optimal_th_rf:.2f})")
print(f"  A lower threshold captures more TPs (more lapsed customers caught),")
print(f"  and the net gain per TP ({EXPECTED_REVENUE - CAMPAIGN_COST:.0f} PLN) justifies some extra FPs ({CAMPAIGN_COST} PLN each)")
print(f"  until the marginal FP cost exceeds the marginal TP gain.")

# ── 8. Lift Analysis ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: LIFT ANALYSIS (CUMULATIVE GAINS CURVE)")
print("=" * 70)

# Sort test customers by RF predicted probability (descending)
sorted_indices = np.argsort(-y_prob_rf)
y_test_sorted = np.asarray(y_test)[sorted_indices]

total_lapsed = y_test_sorted.sum()
cum_lapsed = np.cumsum(y_test_sorted)
pct_contacted = np.arange(1, n_test + 1) / n_test * 100
pct_lapsed_captured = cum_lapsed / total_lapsed * 100

# Print lift at key contact levels
print(f"\nCumulative Gains (RF):")
print(f"  Total lapsed in test set: {total_lapsed}")
for target_pct in [10, 20, 30, 50]:
    idx = int(n_test * target_pct / 100) - 1
    print(f"  At {target_pct}% contacted ({int(n_test * target_pct / 100)} customers): "
          f"{pct_lapsed_captured[idx]:.1f}% of lapsed captured "
          f"({int(cum_lapsed[idx])} out of {total_lapsed})")
    # Also compute lift
    lift = pct_lapsed_captured[idx] / target_pct
    print(f"    Lift = {lift:.2f}x vs random")

# Also do LR for comparison
sorted_indices_lr = np.argsort(-y_prob_lr)
y_test_sorted_lr = np.asarray(y_test)[sorted_indices_lr]
cum_lapsed_lr = np.cumsum(y_test_sorted_lr)
pct_lapsed_captured_lr = cum_lapsed_lr / total_lapsed * 100

print(f"\nCumulative Gains (LR):")
for target_pct in [10, 20, 30, 50]:
    idx = int(n_test * target_pct / 100) - 1
    print(f"  At {target_pct}% contacted: "
          f"{pct_lapsed_captured_lr[idx]:.1f}% of lapsed captured "
          f"({int(cum_lapsed_lr[idx])} out of {total_lapsed})")
    lift = pct_lapsed_captured_lr[idx] / target_pct
    print(f"    Lift = {lift:.2f}x vs random")

# Plot lift curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pct_contacted, pct_lapsed_captured, "r-", label="Random Forest", linewidth=2)
ax.plot(pct_contacted, pct_lapsed_captured_lr, "b-", label="Logistic Regression", linewidth=2)
ax.plot([0, 100], [0, 100], "k--", label="Random (no model)", alpha=0.5)
ax.set_xlabel("% of Customers Contacted")
ax.set_ylabel("% of Lapsed Customers Captured")
ax.set_title("Cumulative Gains (Lift) Curve")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mp4_lift_curve.png", dpi=150)
plt.close()
print(f"\nLift curve saved.")

# ── 9. Annual Profit Estimate ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: ANNUAL PROFIT ESTIMATE")
print("=" * 70)

total_customers = 5000
test_fraction = len(y_test) / total_customers

print(f"\nTest set: {len(y_test)} customers out of {total_customers}")
print(f"Test fraction: {test_fraction:.4f} ({test_fraction * 100:.1f}%)")

# Scale to full base
annual_profit_lr = optimal_profit_lr / test_fraction
annual_profit_rf = optimal_profit_rf / test_fraction
annual_loss_all = profit_all / test_fraction

print(f"\nAnnual profit estimates (extrapolated to full base):")
print(f"  LR (optimal th={optimal_th_lr:.2f}): {annual_profit_lr:.2f} PLN")
print(f"  RF (optimal th={optimal_th_rf:.2f}): {annual_profit_rf:.2f} PLN")
print(f"  Contact Everyone:                     {annual_loss_all:.2f} PLN")
print(f"\n  Savings from using RF model vs Contact Everyone:")
print(f"    {annual_profit_rf - annual_loss_all:.2f} PLN per year")

# ── 10. Save Checkpoint for MP5 ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: SAVING CHECKPOINT FOR MP5")
print("=" * 70)

checkpoint_mp4 = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": feature_names,
    "gender_test": gender_test,
    "lr_model": lr_model,
    "rf_model": rf_model,
    "y_prob_lr": y_prob_lr,
    "y_prob_rf": y_prob_rf,
    "CAMPAIGN_COST": CAMPAIGN_COST,
    "EXPECTED_REVENUE": EXPECTED_REVENUE,
    "optimal_threshold_lr": optimal_th_lr,
    "optimal_profit_lr": optimal_profit_lr,
    "optimal_threshold_rf": optimal_th_rf,
    "optimal_profit_rf": optimal_profit_rf,
}

mp4_ckpt_path = CHECKPOINT_DIR / "mp4_checkpoint.pkl"
with open(mp4_ckpt_path, "wb") as f:
    pickle.dump(checkpoint_mp4, f)
print(f"Checkpoint saved: {mp4_ckpt_path}")

# ── SUMMARY OF KEY VALUES FOR MCQ TEST ────────────────────────────
print("\n" + "=" * 70)
print("🎯 KEY VALUES FOR MCQ TEST")
print("=" * 70)
print(f"  Expected Revenue (median lapsed test spend): {EXPECTED_REVENUE:.2f} PLN")
print(f"  Campaign Cost: {CAMPAIGN_COST} PLN")
print(f"  Net gain per TP: {EXPECTED_REVENUE - CAMPAIGN_COST:.2f} PLN")
print(f"  Loss per FP: {CAMPAIGN_COST} PLN")
print()
print(f"  LR profit at th=0.5: {profit_lr_05:.2f} PLN ({profit_lr_05/n_test:.2f} PLN/record)")
print(f"  RF profit at th=0.5: {profit_rf_05:.2f} PLN ({profit_rf_05/n_test:.2f} PLN/record)")
print()
print(f"  Contact Everyone profit: {profit_all:.2f} PLN ({profit_all/n_test:.2f} PLN/record)")
print(f"  Contact Nobody profit: {profit_none:.2f} PLN")
print()
print(f"  LR optimal threshold: {optimal_th_lr:.2f} → profit: {optimal_profit_lr:.2f} PLN")
print(f"  RF optimal threshold: {optimal_th_rf:.2f} → profit: {optimal_profit_rf:.2f} PLN")
print()

# Print lift at 20% for RF (specifically asked in brief)
idx_20 = int(n_test * 0.20) - 1
lift_rf_20 = pct_lapsed_captured[idx_20] / 20
print(f"  RF Lift at 20% contact: {pct_lapsed_captured[idx_20]:.1f}% captured = {lift_rf_20:.2f}x lift")
print()
print(f"  Annual profit (LR optimal): {annual_profit_lr:.2f} PLN")
print(f"  Annual profit (RF optimal): {annual_profit_rf:.2f} PLN")
print(f"  Annual loss (contact everyone): {annual_loss_all:.2f} PLN")
