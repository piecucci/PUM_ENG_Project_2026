# MP3: Baseline Modeling & Algorithm Comparison

## Scenario

Your data is clean and ready for modeling. MajsterPlus needs to know: **can we
predict which customers will lapse?** You'll train two different algorithms —
Logistic Regression (simple, interpretable) and Random Forest (complex,
powerful) — and compare their performance.

This mini-project covers the **Modeling** phase of CRISP-DM.

## Learning Objectives

By the end of this mini-project, you should be able to:

- Train classification models with scikit-learn
- Evaluate using confusion matrix, report, and ROC-AUC
- Compare models using ROC curves on the same plot
- Analyze feature importance from tree-based models
- Detect overfitting by comparing train vs. test

## What You Receive

- `3. notebooks/mp3_starter.ipynb` — starter with section headers
- Checkpoint from MP2 (or `checkpoint_for_mp3.pkl`)

**If you didn't complete MP2**, load the golden checkpoint:

```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp3.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## What You Do

| Step | Task                                     | Pre-filled? | Time      |
| ---- | ---------------------------------------- | ----------- | --------- |
| 1    | Load checkpoint                          | Yes         | 5 min     |
| 2    | **TODO**: Train LR, evaluate metrics     | **TODO**    | 30 min    |
| 3    | **TODO**: Train RF, evaluate metrics     | **TODO**    | 25 min    |
| 4    | **TODO**: Plot ROC curves (same chart)   | **TODO**    | 15 min    |
| 5    | **TODO**: Visualize RF feature importance | **TODO**    | 15 min    |
| 6    | **TODO**: Compare train vs. test accuracy| **TODO**    | 10 min    |
| 7    | **TODO**: Create summary table           | **TODO**    | 15 min    |
| 8    | Save checkpoint for MP4                  | Yes         | 5 min     |
|      | **Total**                                |             | **~2 h**  |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand

- [ ] How to interpret a confusion matrix in business context
- [ ] What the modest AUC improvement suggests about signal
- [ ] LogisticRegression test accuracy
- [ ] RandomForest recall for class 1 (lapsed)
- [ ] RandomForest's most important feature
- [ ] Train accuracy for both models (overfitting check)

## Hints and Common Pitfalls

1. **Use exact hyperparameters**:
   - `LogisticRegression(random_state=42, max_iter=1000)`
   - `RandomForestClassifier(random_state=42, n_estimators=100)`

2. **Confusion matrix interpretation**: `confusion_matrix(y_test, y_pred)` gives:

   ```text
   [[TN, FP],
    [FN, TP]]
   ```

3. **ROC curves**: Use `RocCurveDisplay.from_predictions()` for plotting.

4. **Feature importance**: `rf.feature_importances_` gives importance for each
   feature in order. Map them to names.

5. **Overfitting check**: Random Forest 100% train accuracy is common. Check
   how much worse it performs on test.

6. **Precision vs. Recall**:
   - **Precision** = TP / (TP + FP) — "Of predicted lapsed, how many truly are?"
   - **Recall** = TP / (TP + FN) — "Of truly lapsed, how many did we catch?"

## Reproducibility

- Random seed: 42 (for all models)
- LogisticRegression: `max_iter=1000`
- RandomForest: `n_estimators=100`
