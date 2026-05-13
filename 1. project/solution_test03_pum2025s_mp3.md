# MP3: Baseline Modeling & Algorithm Comparison - Test & Solutions

## Question 1

A Random Forest achieves 100% accuracy on training but only 81% on the test set. A Logistic Regression achieves 82% on both. Which statement is correct?
- The Random Forest is better because it achieves higher training accuracy
- The Random Forest is overfitting — it memorizes training data but fails to generalize, while Logistic Regression generalizes well
- Both models are underfitting because neither exceeds 85% test accuracy
- The training/test gap for Random Forest is normal and does not indicate a problem

**Answer:** The Random Forest is overfitting — it memorizes training data but fails to generalize, while Logistic Regression generalizes well

**Step-by-step Explanation:**
1. Random Forest achieves 100% training accuracy but only 81% on the test set — a gap of ~19 percentage points. This is a classic sign of **overfitting**: the model memorizes the training data (including noise) but fails to generalize to unseen examples.
2. Logistic Regression achieves ~82% on both train and test, meaning there is virtually no gap. This indicates the model generalizes well — it learned a pattern that holds equally on unseen data.
3. Training accuracy alone is meaningless for model quality; what matters is how performance transfers to the test set. A model that "memorizes" everything (100% train) but drops significantly on test data is overfitting.
4. The brief explicitly warns: "Random Forest 100% train accuracy is common. Check how much worse it performs on test." (Hint #5).

---

## Question 2

For a customer lapse prediction model where the goal is to identify at-risk customers for a reactivation campaign, which metric is most important?
- Accuracy — highest percentage of correct predictions overall
- Precision — we want every contacted customer to actually be lapsed
- Recall — we want to identify as many lapsed customers as possible, even at the cost of some false alarms
- Specificity — we want to correctly identify active customers

**Answer:** Recall — we want to identify as many lapsed customers as possible, even at the cost of some false alarms

**Step-by-step Explanation:**
1. The business goal is a **reactivation campaign** — the marketing team wants to reach as many at-risk (lapsed) customers as possible before they leave permanently.
2. **Recall** = TP / (TP + FN) measures "Of all truly lapsed customers, how many did we catch?" This directly maps to the campaign objective.
3. Missing a lapsed customer (False Negative) means a lost reactivation opportunity — a real business cost. Contacting an active customer by mistake (False Positive) wastes some campaign budget, but the cost is lower.
4. **Accuracy** is misleading on imbalanced datasets (only ~19.5% are lapsed). **Precision** prioritizes avoiding false alarms over catching all at-risk customers. **Specificity** focuses on correctly classifying active customers, which is less relevant to the reactivation campaign goal.

---

## Question 3

When comparing two classifiers using ROC curves, how do you determine which model is better?
- The model with the steeper initial slope is always better
- The model whose curve is closest to the diagonal line performs best
- The model with the larger Area Under the Curve (AUC) has better overall discrimination ability
- The model that reaches 100% True Positive Rate first is always superior

**Answer:** The model with the larger Area Under the Curve (AUC) has better overall discrimination ability

**Step-by-step Explanation:**
1. The ROC curve plots True Positive Rate (sensitivity) vs. False Positive Rate (1 − specificity) at all classification thresholds.
2. The **Area Under the Curve (AUC)** summarizes the entire ROC curve into a single number between 0 and 1. A higher AUC means the model has better discrimination ability across all thresholds.
3. A curve **closest to the diagonal** (AUC ≈ 0.5) represents a random classifier — this is the **worst**, not the best.
4. Steeper initial slope or reaching 100% TPR first can be useful in specific threshold regions, but neither is a holistic measure of overall model quality. AUC captures the full picture.

---

## Question 4

You observe that Random Forest ranks `days_since_last_purchase` as the most important feature. Given that `is_lapsed` is defined as "no purchase in last 90 days," should this feature be included?
- Yes — it's the most predictive feature and should always be included
- It should be used cautiously — since the target is partly derived from this feature, including it may give artificially inflated performance
- No — it should be removed entirely because it directly determines the target
- Yes, but only if its importance exceeds 50% of total feature importance

**Answer:** It should be used cautiously — since the target is partly derived from this feature, including it may give artificially inflated performance

**Step-by-step Explanation:**
1. The target variable `is_lapsed` is defined as "no purchase in the last 90 days." The feature `days_since_last_purchase` measures exactly the same underlying concept — recency of the last purchase.
2. This creates a form of **data leakage**: the feature is partially (or strongly) derived from the same information used to define the target. The model doesn't truly "predict" lapse — it learns a near-tautological mapping.
3. However, the relationship is not perfectly deterministic. `days_since_last_purchase` is a continuous value while `is_lapsed` is binary, and the 90-day threshold introduces a boundary. So it's not a direct 1:1 mapping — hence "cautiously" rather than "remove entirely."
4. In practice, such features inflate metrics (accuracy, AUC) and give a false sense of model performance. They should be flagged and possibly removed in production models.

---

## Question 5

A confusion matrix for a lapse prediction model shows: TN=757, FP=44, FN=125, TP=70. Which statement correctly interprets this in the MajsterPlus business context?
- The model correctly identified 70 lapsed customers but missed 125 — the marketing team would reach only 36% of at-risk customers
- The model has 44 false positives, meaning it is too aggressive and should not be deployed
- The model's accuracy is 70/(70+44) = 61%, which is poor
- The 757 true negatives are the most important metric — correctly identifying active customers saves money

**Answer:** The model correctly identified 70 lapsed customers but missed 125 — the marketing team would reach only 36% of at-risk customers

**Step-by-step Explanation:**
1. From the confusion matrix: TP = 70 (correctly predicted lapsed), FN = 125 (lapsed but predicted active).
2. **Recall** for the lapsed class = TP / (TP + FN) = 70 / (70 + 125) = 70 / 195 ≈ **0.359 ≈ 36%**. This means the marketing team would only reach ~36% of truly at-risk customers.
3. The statement about accuracy being 70/(70+44) = 61% is **wrong** — that formula calculates precision, not accuracy. True accuracy = (TN + TP) / total = (757 + 70) / 996 ≈ 83%.
4. 44 false positives is relatively low and does not justify refusing deployment — some false alarms are acceptable in a reactivation campaign.
5. True negatives (active customers) are passively correct — they require no action and are not the priority metric for a campaign targeting at-risk customers.

---

## Question 6

The MP1 baseline LogisticRegression (6 raw numeric features, no cleaning) achieved ROC-AUC ≈ 0.83. After full data preparation in MP2 (cleaning, encoding, scaling), the MP3 LogisticRegression achieves ROC-AUC ≈ 0.84. What does this modest improvement suggest?
- Data preparation was unnecessary — the raw features were already sufficient
- Most of the predictive signal was already present in the raw numeric features; data preparation added marginal improvement through better feature representation
- The MP3 model is overfitting because its AUC barely changed despite adding many more features
- StandardScaler was applied incorrectly — proper scaling should improve AUC by at least 0.10

**Answer:** Most of the predictive signal was already present in the raw numeric features; data preparation added marginal improvement through better feature representation

**Step-by-step Explanation:**
1. The MP1 baseline achieved AUC ≈ 0.83 using only 6 raw numeric features with no cleaning. The MP3 model achieves AUC ≈ 0.84 after full data preparation (cleaning, encoding, scaling) with 37 features.
2. The improvement of ~0.01 AUC is modest, suggesting that the core predictive signal (e.g., `days_since_last_purchase`, `purchase_count`, `total_spend`) was already captured by the raw numeric features.
3. Data preparation was **not unnecessary** — it improved feature quality, handled missing values, encoded categoricals, and made the pipeline production-ready. But it did not dramatically boost AUC because the raw signal was already strong.
4. A barely-changed AUC does **not** indicate overfitting — overfitting is indicated by a train/test gap, not by a plateau in performance. And there is no rule that scaling must improve AUC by 0.10.

---

## Question 7

What is the Random Forest's recall (sensitivity) for the lapsed class (class 1) on the test set?
- 0.19
- 0.27
- 0.38
- 0.53

**Answer:** 0.38

**Step-by-step Explanation:**
1. The Random Forest is trained with `RandomForestClassifier(random_state=42, n_estimators=100)` on the MP2 checkpoint data.
2. Recall for class 1 (lapsed) = TP / (TP + FN) from the RF confusion matrix on the test set.
3. With the standard hyperparameters and the MP2 prepared dataset, the Random Forest achieves a recall of approximately **0.38** for the lapsed class — meaning it catches about 38% of truly lapsed customers.
4. This is notably low for a reactivation campaign, and highlights the trade-off the RF makes: it has relatively high precision but low recall for the minority class.

---

## Question 8

Which feature has the highest importance in the Random Forest model?
- total_spend
- avg_basket_value
- days_since_last_purchase
- store_distance_km

**Answer:** days_since_last_purchase

**Step-by-step Explanation:**
1. Random Forest feature importance is obtained via `rf.feature_importances_`, which measures the average impurity decrease contributed by each feature across all trees.
2. `days_since_last_purchase` dominates because of its strong relationship with the target variable `is_lapsed` (defined as "no purchase in last 90 days"). The feature nearly encodes the target definition.
3. The brief explicitly warns about this in Hint #4: "rf.feature_importances_ gives importance for each feature in order. Map them to names." And Q4 of this test directly discusses the implications of this top feature.
4. While `total_spend` and `avg_basket_value` are predictive, they carry much less direct information about the lapse definition than recency of purchase.

---

## Question 9

Comparing confusion matrices, the LR has 44 false positives while the RF has 66 false positives. What does this mean in the business context?
- LR would waste campaign budget on fewer active customers — it is more conservative in predicting lapse
- LR identifies more lapsed customers than RF
- RF has better precision because it contacts more customers
- The difference is negligible and both models perform equally

**Answer:** LR would waste campaign budget on fewer active customers — it is more conservative in predicting lapse

**Step-by-step Explanation:**
1. **False positives** are active customers incorrectly predicted as lapsed. In the campaign context, these are customers who would receive a reactivation offer unnecessarily — wasted marketing budget.
2. LR has 44 FP vs. RF's 66 FP. This means LR incorrectly contacts **22 fewer** active customers, making it more **conservative** (less aggressive) in predicting lapse.
3. "LR identifies more lapsed customers than RF" is a claim about recall/TP, not about FP — and isn't necessarily true based on FP counts alone.
4. RF having more FP means **lower** precision (not better), because more of its positive predictions are wrong.
5. A difference of 22 false positives (44 vs. 66, a ~50% relative increase) is meaningful, not negligible.

---

## Question 10

What is the Logistic Regression's test accuracy in MP3?
- 0.78
- 0.83
- 0.87
- 0.92

**Answer:** 0.83

**Step-by-step Explanation:**
1. The Logistic Regression is trained with `LogisticRegression(random_state=42, max_iter=1000)` on the full MP2 cleaned/encoded/scaled feature set.
2. Test accuracy is computed as `accuracy_score(y_test, y_pred_lr)` or equivalently `lr.score(X_test, y_test)`.
3. With ~80.5% of customers being active, even a moderate model achieves around this baseline. The LR, with its learned coefficients on the 37-feature set, achieves **~0.83** test accuracy.
4. This is consistent with the brief's hints: "LogisticRegression test accuracy" is listed as something you should know, and the notebook's overfitting section notes LR achieves ~82% on both train and test (which rounds to 0.83 given 4-decimal rounding in the classification report).

---

## References

All answers are based on:
- **Brief**: `1. project/mp3_brief.md`
- **Starter notebook**: `3. notebooks/mp3_starter.ipynb`
- **Hyperparameters**: `LogisticRegression(random_state=42, max_iter=1000)`, `RandomForestClassifier(random_state=42, n_estimators=100)`
- **Dataset**: MP2 checkpoint (4,978 samples after outlier removal, 37 features after one-hot encoding, 80/20 stratified split)

---

*Solution for Test 03 PUM2025S - Mini Project 3*
*Generated based on MP3 brief, starter notebook, and MP2 data pipeline*
