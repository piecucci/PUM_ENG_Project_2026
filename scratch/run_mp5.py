import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load checkpoint
checkpoint_path = "/home/mp/PUM_ENG_Project_2026/2. data/checkpoints/checkpoint_for_mp5.pkl"
with open(checkpoint_path, "rb") as f:
    checkpoint = pickle.load(f)

X_train = checkpoint["X_train"]
X_test = checkpoint["X_test"]
y_train = checkpoint["y_train"]
y_test = checkpoint["y_test"]
gender_test = checkpoint["gender_test"]
lr_model = checkpoint["lr_model"]
rf_model = checkpoint["rf_model"]
y_prob_lr = checkpoint["y_prob_lr"]
y_prob_rf = checkpoint["y_prob_rf"]

CAMPAIGN_COST = checkpoint.get("CAMPAIGN_COST", 80)
EXPECTED_REVENUE = checkpoint.get("EXPECTED_REVENUE", 140.0)

# Train Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_prob_gb = gb.predict_proba(X_test)[:, 1]
y_pred_gb = gb.predict(X_test)

# Train Voting Classifier (soft voting, LR, RF, GB)
# Since the Voting Classifier needs already fitted estimators, or we can use VotingClassifier on them directly.
# Wait, let's create a VotingClassifier and fit it, or define estimators and fit them.
# The brief says: "Include LR, RF, and GB as estimators. Use voting='soft'"
# Let's check how LR and RF are loaded. Are they already fitted? Yes, lr_model and rf_model are.
# But for VotingClassifier, we usually fit the ensemble model on X_train, y_train.
estimators = [('lr', lr_model), ('rf', rf_model), ('gb', gb)]
vc = VotingClassifier(estimators=estimators, voting='soft')
vc.fit(X_train, y_train)
y_prob_vc = vc.predict_proba(X_test)[:, 1]
y_pred_vc = vc.predict(X_test)

# Calculate ROC-AUCs
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)
auc_gb = roc_auc_score(y_test, y_prob_gb)
auc_vc = roc_auc_score(y_test, y_prob_vc)

print("=== ROC-AUC Scores ===")
print(f"Logistic Regression: {auc_lr:.4f}")
print(f"Random Forest:       {auc_rf:.4f}")
print(f"Gradient Boosting:   {auc_gb:.4f}")
print(f"Voting Classifier:   {auc_vc:.4f}")

# Profit function
def compute_profit(y_true, y_pred, revenue=EXPECTED_REVENUE, cost=CAMPAIGN_COST):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp * (revenue - cost) + fp * (-cost)

print("\n=== Profit at default threshold 0.5 ===")
y_pred_lr = (y_prob_lr >= 0.5).astype(int)
y_pred_rf = (y_prob_rf >= 0.5).astype(int)
print(f"Logistic Regression: {compute_profit(y_test, y_pred_lr):.1f} PLN")
print(f"Random Forest:       {compute_profit(y_test, y_pred_rf):.1f} PLN")
print(f"Gradient Boosting:   {compute_profit(y_test, y_pred_gb):.1f} PLN")
print(f"Voting Classifier:   {compute_profit(y_test, y_pred_vc):.1f} PLN")

# Sweep thresholds for optimal profit
thresholds = np.arange(0.05, 1.0, 0.05)
print("\n=== Optimal Profit and Threshold Sweep ===")
for name, probs in [("LR", y_prob_lr), ("RF", y_prob_rf), ("GB", y_prob_gb), ("VC", y_prob_vc)]:
    best_profit = -np.inf
    best_th = None
    for th in thresholds:
        preds = (probs >= th).astype(int)
        profit = compute_profit(y_test, preds)
        if profit > best_profit:
            best_profit = profit
            best_th = th
    print(f"{name:2}: Best Profit = {best_profit:.1f} PLN at Threshold = {best_th:.2f}")

# Fairness Analysis
print("\n=== Fairness Analysis (Recall by Gender M/K) ===")
# Split test set by gender_test
# gender_test is a pandas series
gender_vals = gender_test.values if hasattr(gender_test, 'values') else gender_test
for name, probs in [("LR", y_prob_lr), ("RF", y_prob_rf), ("GB", y_prob_gb), ("VC", y_prob_vc)]:
    for th in [0.5, "optimal"]:
        # Find optimal threshold first for this model
        if th == "optimal":
            best_profit = -np.inf
            best_th = 0.5
            for t in thresholds:
                preds = (probs >= t).astype(int)
                profit = compute_profit(y_test, preds)
                if profit > best_profit:
                    best_profit = profit
                    best_th = t
            t_val = best_th
        else:
            t_val = th
        
        preds = (probs >= t_val).astype(int)
        
        # Split by gender
        mask_m = (gender_vals == 'M')
        mask_k = (gender_vals == 'K')
        
        recall_m = recall_score(y_test[mask_m], preds[mask_m])
        recall_k = recall_score(y_test[mask_k], preds[mask_k])
        precision_m = precision_score(y_test[mask_m], preds[mask_m], zero_division=0)
        precision_k = precision_score(y_test[mask_k], preds[mask_k], zero_division=0)
        
        print(f"{name} (th={t_val:.2f}): M Recall={recall_m:.4f}, K Recall={recall_k:.4f}, Gap={abs(recall_m - recall_k):.4f} | M Prec={precision_m:.4f}, K Prec={precision_k:.4f}")
