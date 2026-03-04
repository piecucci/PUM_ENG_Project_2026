# MP5: Model Comparison & Final Recommendation

## Scenario

You've built and evaluated two models. Now MajsterPlus asks for your **final
recommendation**: Which model should we deploy? At what threshold? What about
fairness across customer demographics? The VP of Marketing needs a written
recommendation she can present to the board.

This mini-project covers the **Evaluation** phase of CRISP-DM — synthesis,
comparison, and recommendation.

## Learning Objectives

By the end of this mini-project, you should be able to:

- Train and evaluate additional algorithms (GradientBoosting, VotingClassifier)
- Compare models across multiple criteria (profit, fairness, interpretability)
- Assess model fairness by analyzing performance across demographic subgroups
- Evaluate model interpretability (coefficients vs. feature importances)
- Write a structured, evidence-based business recommendation

## What You Receive

- `3. notebooks/mp5_starter.ipynb` — minimal scaffolding
- Checkpoint from MP4 (or `checkpoint_for_mp5.pkl`)

**If you didn't complete MP4**, load the golden checkpoint:

```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp5.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## What You Do

| Step | Task                                     | Pre-filled? | Time      |
| ---- | ---------------------------------------- | ----------- | --------- |
| 1    | Load checkpoint                          | Yes         | 5 min     |
| 2-3  | Reuse LR + RF models from checkpoint     | Yes         | 5 min     |
| 4    | **TODO**: Train GradientBoosting         | **TODO**    | 15 min    |
| 5    | **TODO**: Create VotingClassifier        | **TODO**    | 15 min    |
| 6    | **TODO**: Multi-criteria comparison table | **TODO**    | 20 min    |
| 7    | **TODO**: Business profit comparison     | **TODO**    | 20 min    |
| 8    | **TODO**: Fairness analysis              | **TODO**    | 20 min    |
| 9    | **TODO**: Interpretability assessment    | **TODO**    | 15 min    |
| 10   | **TODO**: Write final recommendation     | **TODO**    | 20 min    |
|      | **Total**                                |             | **~2.5 h**|

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand

- [ ] Why the model with highest ROC-AUC might not have highest profit
- [ ] What additional criteria guide selection when AUCs are near-identical
- [ ] VotingClassifier ROC-AUC and which model has the highest ROC-AUC overall
- [ ] Recall gap between gender groups for GradientBoosting
- [ ] Which model you would recommend for deployment and why

## Hints and Common Pitfalls

1. **GradientBoosting**: Use `GradientBoostingClassifier(random_state=42)` with
   default hyperparameters.

2. **VotingClassifier**: Use `voting="soft"` to average probabilities. Include
   LR, RF, and GB as estimators.

3. **Best ROC-AUC ≠ best business model.** The model with the highest AUC may
   not generate the most profit. Compare at each model's optimal threshold.

4. **Fairness analysis**: Use `gender_test` from the checkpoint to split
   predictions by gender (M/K) and compute metrics for each subgroup.

5. **Interpretability ranking**:
   - Logistic Regression: most interpretable (direct coefficients)
   - GradientBoosting/RandomForest: feature importances (what, but not how)
   - VotingClassifier: least interpretable (combines multiple models)

6. **The "simpler model wins" pattern**: Don't be surprised if LogisticRegression
   generates the highest profit.

## Reproducibility

- Random seed: 42 (for GradientBoosting)
- VotingClassifier: soft voting, estimators = [LR, RF, GB]
- Same cost matrix as MP4 (campaign cost = 80 PLN)
