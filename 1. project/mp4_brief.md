# MP4: Model Evaluation & Business Impact

## Scenario

You have trained two models. But MajsterPlus doesn't care about ROC-AUC — they
care about **money**. The marketing VP asks: "If we run a reactivation campaign
targeting your model's predictions, how much profit will we make? And should we
contact everyone, or just the high-risk customers?"

This mini-project covers the **Evaluation** phase of CRISP-DM — translating
statistical metrics into business value.

## Learning Objectives

By the end of this mini-project, you should be able to:

- Construct a cost matrix mapping predictions to financial outcomes
- Calculate profit per record at a given classification threshold
- Compare model-based targeting against naive strategies
- Optimize the classification threshold for maximum profit
- Interpret cumulative gains (lift) curves

## What You Receive

- `3. notebooks/mp4_starter.ipynb` — starter notebook with business context
- Checkpoint from MP3 (or `checkpoint_for_mp4.pkl`)

**If you didn't complete MP3**, load the golden checkpoint:

```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp4.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## Business Parameters

| Parameter                   | Value                                     |
| --------------------------- | ----------------------------------------- |
| Campaign cost per customer  | **80 PLN** (voucher + operations)         |
| Voucher value               | 50 PLN                                    |
| Expected revenue            | **Median total_spend of lapsed test set** |

### Cost matrix

|                                | Actually Lapsed       | Actually Active |
| ------------------------------ | --------------------- | --------------- |
| **Contacted** (pred. lapsed)   | Revenue − 80 PLN (TP) | −80 PLN (FP)    |
| **Not contacted** (pred. act.) | 0 PLN (FN)            | 0 PLN (TN)      |

## What You Do

| Step | Task                                     | Pre-filled? | Time      |
| ---- | ---------------------------------------- | ----------- | --------- |
| 1    | Load checkpoint                          | Yes         | 5 min     |
| 2    | Set up business parameters               | Partially   | 10 min    |
| 3    | **TODO**: Define compute_profit()        | **TODO**    | 15 min    |
| 4    | **TODO**: Calculate profit (th=0.5)      | **TODO**    | 15 min    |
| 5    | **TODO**: Calculate baseline profits     | **TODO**    | 10 min    |
| 6    | **TODO**: Threshold optimization         | **TODO**    | 25 min    |
| 7    | **TODO**: Identify optimal threshold     | **TODO**    | 10 min    |
| 8    | **TODO**: Create lift curve              | **TODO**    | 20 min    |
| 9    | **TODO**: Estimate annual profit         | **TODO**    | 10 min    |
| 10   | Save checkpoint for MP5                  | Yes         | 5 min     |
|      | **Total**                                |             | **~2 h**  |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand

- [ ] Expected revenue per reactivation (median lapsed test spend)
- [ ] Profit from "contact everyone" strategy on the test set
- [ ] LogisticRegression profit at threshold 0.5
- [ ] Why a threshold below 0.5 can be optimal
- [ ] Lift at the 20% contact level (RF cumulative gains curve)

## Hints and Common Pitfalls

1. **Expected revenue**: Use the **median** total_spend of lapsed test
   customers. Load raw data for original values (checkpoint is scaled).

2. **The "contact everyone" strategy loses money.** This is realistic —
   contacting the 80% of active customers is very expensive.

3. **Threshold = 0.5 is NOT optimal.** The whole point of this MP is to find a
   better threshold. Plot profit vs. threshold and find the peak.

4. **Understand threshold mechanics.** The MCQs test why the optimal threshold
   differs from 0.5 and the profit trade-off at 0.5.

5. **Lift curve**: Plot cumulative % of lapsed customers captured vs. % of
   customers contacted. Closer to upper-left is better.

6. **Annual profit extrapolation**: Scale proportionally: `annual = test_profit
   / test_fraction`. (Test set is ~20% of base).

7. **Negative profit is possible** and realistic. If false positives outnumber
   true positives, the campaign loses money.

## Reproducibility

- Campaign cost: 80 PLN (fixed)
- Expected revenue: median of lapsed test customers' total_spend
- Thresholds: np.arange(0.05, 1.0, 0.05) — 19 steps
