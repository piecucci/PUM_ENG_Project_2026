# MP2: Data Cleaning & Feature Engineering

## Scenario

You've explored the MajsterPlus data and identified several quality issues:
Polish date formats, currency strings, missing values, outliers, and impossible
values. Now you need to **clean and transform** the data into a format suitable
for machine learning.

This mini-project covers the **Data Preparation** phase of CRISP-DM — the most
labor-intensive phase in any data science project.

## Learning Objectives

By the end of this mini-project, you should be able to:

- Parse non-standard date formats and currency strings
- Identify and handle different types of missing data
- Detect and remove outliers using the IQR method
- Encode categorical variables (binary, ordinal, one-hot)
- Apply feature scaling (StandardScaler) with proper separation

## What You Receive

- `3. notebooks/mp2_starter.ipynb` — starter notebook with 12-step pipeline
- `2. data/customers.csv` — raw data (or use `checkpoint_for_mp2.csv`)
- `2. data/data_dictionary.md` — documents all 10 data quality issues

## What You Do

You must follow the **mandatory 12-step preprocessing order**.

| Step  | Task                                     | Pre-filled? | Time      |
| ----- | ---------------------------------------- | ----------- | --------- |
| 1-2   | Load data & verify MD5                   | Yes         | 5 min     |
| 3     | Separate target, drop ID                 | Yes         | 5 min     |
| 4     | Parse Polish dates                       | **TODO**    | 15 min    |
| 5     | Clean total_spend (string → float)       | **TODO**    | 10 min    |
| 6     | Replace impossible scores with NaN       | **TODO**    | 10 min    |
| 7     | Impute missing (median/mode)             | **TODO**    | 20 min    |
| 8     | Remove IQR outliers                      | **TODO**    | 15 min    |
| 9     | Encode (binary, ordinal, one-hot)        | **TODO**    | 25 min    |
| 10    | Null assertion gate                      | Yes         | 2 min     |
| 11    | Train/test split (80/20)                 | Yes         | 5 min     |
| 12    | StandardScaler (fit on train)            | **TODO**    | 10 min    |
| Bonus | K-Means clustering                       | **TODO**    | 15 min    |
| Save  | Checkpoint for MP3                       | Yes         | 2 min     |
|       | **Total**                                |             | **~2.5 h**|

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand

- [ ] What mean vs. median of `total_spend` tells you about distribution
- [ ] Why replace impossible scores with NaN *before* median imputation
- [ ] Why filtering `y` and `gender_series` after IQR removal is critical
- [ ] The trade-off between `drop_first=False` and `drop_first=True`
- [ ] Training set dimensions (rows × features)

## Hints and Common Pitfalls

1. **Follow the step order exactly.** Step 6 (impossible values → NaN) must
   happen BEFORE Step 7 (imputation).

2. **Polish date helper is provided** — apply it with
   `.apply(parse_polish_date)`.

3. **total_spend conversion**: Strip "PLN ", then remove commas, then float.

4. **Satisfaction score**: Valid range is [1.0, 5.0]. Outside this range (0.0,
   7.2, -1.0) replace with NaN, then impute.

5. **IQR outlier removal**: Apply the filter. **Don't forget** to filter `y` and
   `gender_series` to keep them aligned.

6. **Encoding order matters**:
   - First: `loyalty_member` → binary (Tak=1, Nie=0)
   - Then: `monthly_income_bracket` → ordinal (A=1, B=2, ..., E=5)
   - Finally: remaining → `pd.get_dummies(drop_first=False)`
   - **Sort columns alphabetically** after encoding.

7. **`drop_first=False`** — keep all dummy columns for pedagogical clarity.

8. **Boolean columns**: Convert to int: `df[bool_cols] = df[bool_cols].astype(int)`

9. **StandardScaler**: Fit on training data ONLY. Fitting on full data = leakage.

## Reproducibility

- Random seed: 42
- Preprocessing order: Steps 1-12 as specified
- `pd.get_dummies(drop_first=False)` — keep all dummies
- Columns sorted alphabetically after encoding
- StandardScaler fit on train only
