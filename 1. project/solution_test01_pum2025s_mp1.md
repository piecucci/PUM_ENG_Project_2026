# Test 01 PUM2025S - Mini Project 1: Solution

## MajsterPlus Customer Lapse Prediction

This document contains the solutions to the 10 questions from Test 01, based on the execution results of the MP1 notebook (`mp1_starter_s32484_05032026.ipynb`).

---

## Question 1: Misleading 95% Accuracy on Imbalanced Data

**Answer: Because 95% accuracy can be achieved by a naive model that predicts all samples as the majority class**

### Explanation:
In the notebook (Section 4: Target Variable & Class Imbalance), the output shows:
```
is_lapsed
0    0.805
1    0.195
Name: proportion, dtype: float64

Lapse rate: 19.5%
```

With only 3% positive samples, a naive model predicting all as negative would achieve 97% accuracy—without learning any useful pattern. This demonstrates why accuracy is misleading on imbalanced datasets.

---

## Question 2: Highly Correlated Features (`total_spend` with `purchase_count × avg_basket_value`)

**Answer: Investigate whether one column is derived from the others, and consider dropping the redundant one to reduce multicollinearity**

### Explanation:
In the notebook (Section 2: First Look at the Data), we observe:
- `total_spend` is stored as a string ("PLN 1,496.76")
- `purchase_count` and `avg_basket_value` are numeric columns
- Mathematically: `total_spend ≈ purchase_count × avg_basket_value`

This is a clear case of multicollinearity in CRISP-DM's Data Preparation phase. The redundant column should be investigated and possibly dropped to improve model stability.

---

## Question 3: Next Step After Discovering Data Quality Issues

**Answer: Clean, transform, and prepare the data so it is suitable for modeling**

### Explanation:
According to CRISP-DM, after Data Understanding (exploring and discovering issues like missing values, Polish dates, currency strings), the next phase is **Data Preparation**.

The notebook shows exactly these issues:
- `registration_date`: "21-kwi-2022" (Polish month abbreviations)
- `total_spend`: "PLN 1,496.76" (currency strings)
- Missing values in 5 columns (see Section 3)

---

## Question 4: Polish Date Parsing Error

**Answer: The month abbreviations are in Polish (e.g., "sty" = January), which pandas cannot parse automatically**

### Explanation:
From the notebook output in Section 2:
```
registration_date: 21-kwi-2022, 30-kwi-2024, 23-gru-2023, 04-maj-2023
```

Polish month abbreviations:
- "sty" = January (styczeń)
- "kwi" = April (kwiecień)
- "gru" = December (grudzień)
- "maj" = May (maj)

Pandas' default `to_datetime()` cannot parse these without specifying a custom format or locale.

---

## Question 5: Why Split Data Before Feature Scaling

**Answer: Because fitting the scaler on the full dataset leaks information about the test set into the training process (data leakage)**

### Explanation:
`StandardScaler` computes mean and standard deviation from the entire dataset. If applied before the train/test split:
1. Test set statistics "leak" into the training process
2. The model sees information it shouldn't have during training
3. Performance metrics become artificially inflated

**Correct approach**: Fit the scaler on training data only, then transform both training and test sets using those statistics.

---

## Question 6: Dataset Ratio Interpretation (5000 customers, ~25000 transactions)

**Answer: Each customer has on average about 5 transactions, suggesting moderate purchase frequency**

### Explanation:
From notebook Section 2b (Dataset Relationship Analysis):
```
Customers shape (5000, 21)
Transactions shape: (24943, 8)

Average transactions per customer: 4.99
Interpretation: On average, each customer has ~5 transactions, indicating moderate purchase frequency.
```

This ratio (25000/5000 ≈ 5) represents the average number of transactions per customer.

---

## Question 7: Naive Model Predicting All Active

**Answer: 80.5% — it correctly classifies all active customers but misses every lapsed one**

### Explanation:
From the notebook (Section 4):
```
Lapse rate: 19.5%
```

If we predict all customers as "active" (is_lapsed = 0):
- 80.5% of customers (the active ones) → correct predictions
- 19.5% (lapsed customers) → incorrect predictions

Accuracy = 80.5%, but the model has zero predictive power for the positive class.

---

## Question 8: Most Concerning Missing Column

**Answer: `monthly_income_bracket`, because at 641 missing (12.8%) it has the highest missing rate, and income likely correlates with purchasing behavior (MAR risk)**

### Explanation:
From notebook Section 3 (Missing Value Analysis):
```
                        Missing Count  Missing %
monthly_income_bracket            641      12.82
online_ratio                      429       8.58
satisfaction_score                296       5.92
referral_source                   190       3.80
age                               137       2.74
```

This column has:
- Highest missing rate (12.82%)
- Strong potential for MAR (Missing At Random) - customers who haven't purchased much may not have income data
- Income likely correlates with purchasing behavior and target variable

---

## Question 9: Interpreting Extreme `avg_basket_value` Values

**Answer: These are likely data entry errors or exceptional bulk purchases — they are extreme outliers that could distort model training, particularly for distance-based or linear models**

### Explanation:
From notebook Section 2 (Statistical Summary):
```
avg_basket_value:
mean: 245.05
50% (median): 219.39
max: 11994.31
```

With mean ~280 PLN, median ~250 PLN, values exceeding 5,000 PLN are ~17-20x the median. These extreme outliers can:
- Skew mean-based models (linear regression)
- Distort distance-based algorithms (KNN, SVM)
- Need investigation or handling (e.g., capping, removal)

---

## Question 10: Baseline LogisticRegression ROC-AUC

**Answer: ~0.83**

### Explanation:
The notebook explicitly shows the baseline LogisticRegression results:
- Output shows: `Accuracy: 0.8348`
- Line 1126 states: `5. Baseline ROC-AUC: ~0.83 indicates good learnability`
- mp3_starter.ipynb confirms: "In MP1, the baseline LogisticRegression achieved ROC-AUC ≈ 0.83"

This is impressive - even with 6 raw numeric features, no data cleaning, and no feature engineering, the baseline achieves ~0.83 ROC-AUC, indicating strong predictive signal in the customer data.

---

## References

All answers are based on the execution results of:
- **File**: `3. notebooks/mp1_starter_s32484_05032026.ipynb`
- **Sections**:
  - Section 0: Setup & Reproducibility
  - Section 1: Data Loading & Verification
  - Section 2: First Look at the Data
  - Section 2b: Dataset Relationship Analysis
  - Section 3: Missing Value Analysis
  - Section 4: Target Variable & Class Imbalance

---

*Solution for Test 01 PUM2025S - Mini Project 1*
*Generated based on notebook execution outputs*