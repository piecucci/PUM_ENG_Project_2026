# MP5: Model Comparison & Final Recommendation - Solutions

This document provides step-by-step solutions and calculations for the 10 questions in Test 5 based on the MP5 checkpoint.

---

## Quick Answer Key

| Question | Correct Option / Answer | Key Rationale / Calculation |
| :--- | :--- | :--- |
| **Q1** | The model with the best statistical metric may not produce the highest business profit — threshold optimization and cost matrix analysis should also be considered | Business profit is determined by the decision threshold and cost matrix asymmetry. |
| **Q2** | Model B, because the 0.35 gap in recall means the model is much better at identifying at-risk customers in one gender group | Fairness is assessed using demographic subgroups metrics. A gap of 0.35 is highly significant compared to Model A's 0.02. |
| **Q3** | 1, 2, 3 | Logistic Regression (coefficients) > Random Forest (global feature importances) > Voting Classifier (black-box ensemble). |
| **Q4** | The class probabilities from each model are averaged, and the class with the highest average probability is predicted | Definition of "soft" voting in scikit-learn's `VotingClassifier`. |
| **Q5** | Concept drift — customer behavior may change over time, causing the model's assumptions to become invalid | Customer behavioral patterns naturally shift over time (temporal drift). |
| **Q6** | VotingClassifier (≈0.85) | Exact test ROC-AUC: VC (0.8508) > GB (0.8495) > RF (0.8457) > LR (0.8352). |
| **Q7** | ROC-AUC measures discrimination across all thresholds, but profit depends on predictions at one specific threshold — a model that is well-calibrated in the high-precision region can outperform one with better overall AUC | AUC integrates performance over all thresholds, whereas profit is computed at a single operational threshold. |
| **Q8** | Consider business profit at the optimal threshold, model interpretability, fairness across demographic groups, and deployment complexity | When statistical metrics are near-identical, selection must prioritize business utility, ease of maintenance, transparency, and fairness. |
| **Q9** | The model treats both gender groups approximately equally in terms of identifying at-risk customers | A recall gap of under 2% (0.0161) shows negligible difference in targeting performance between genders M and K. |
| **Q10**| “LogisticRegression produces the highest campaign profit, its coefficients directly show which factors drive lapse, and it treats gender groups fairly — making it the safest choice to explain to the board and to regulators” | LR generates the highest business profit (440 PLN at th=0.45), is fully interpretable, and has excellent fairness metrics. |

---

## Detailed Step-by-Step Solutions

### Question 1
**Answer:** The model with the best statistical metric may not produce the highest business profit — threshold optimization and cost matrix analysis should also be considered.

**Explanation:**
- **Why:** In real-world business scenarios, the cost of a False Positive (e.g., spending campaign budget on an active customer) and a False Negative (missing an at-risk customer) are highly asymmetric.
- **Example from calculations:**
  - Gradient Boosting achieves a higher ROC-AUC (0.8495) than Logistic Regression (0.8352).
  - However, at the default threshold of 0.5:
    - Logistic Regression makes **+380.0 PLN** profit.
    - Gradient Boosting makes **-980.0 PLN** profit (a loss).
  - This demonstrates that standard statistical metrics like ROC-AUC do not map directly to profit without threshold optimization.

---

### Question 2
**Answer:** Model B, because the 0.35 gap in recall means the model is much better at identifying at-risk customers in one gender group.

**Explanation:**
- **Why:** In algorithmic fairness, we look at the difference in performance metrics (like recall or selection rate) across protected attributes (e.g., gender).
- **Calculation:**
  - **Model A Recall Gap:** $|0.40 - 0.38| = 0.02$ (2% difference, very fair)
  - **Model B Recall Gap:** $|0.60 - 0.25| = 0.35$ (35% difference, significant fairness concern)
  - A recall gap larger than $0.05$ (5%) is generally considered problematic in regulatory contexts.

---

### Question 3
**Answer:** 1, 2, 3

**Explanation:**
1. **Logistic Regression (1 - Most Interpretable):** Coefficients represent log-odds directly, providing a clear directional impact and magnitude for each feature.
2. **Random Forest (2 - Moderately Interpretable):** Provides feature importances representing how much each feature split contributed to reducing impurity. We know *which* features matter, but not the direction of their impact without further tools.
3. **Voting Classifier (3 - Least Interpretable):** Combining multiple distinct algorithms (linear model, bag of trees, boosted trees) results in a complex black-box model whose individual decisions cannot be easily traced back to a single model parameter.

---

### Question 4
**Answer:** The class probabilities from each model are averaged, and the class with the highest average probability is predicted.

**Explanation:**
- **Soft Voting:** $\hat{p}(y=1|x) = \frac{p_{LR} + p_{RF} + p_{GB}}{3}$. The final class is predicted based on whether this average probability exceeds the classification threshold (usually 0.5).
- **Hard Voting:** In contrast, hard voting uses majority rule on the final class predictions ($0$ or $1$).

---

### Question 5
**Answer:** Concept drift — customer behavior may change over time, causing the model's assumptions to become invalid.

**Explanation:**
- **Concept Drift:** Reflects shifts in the relationship between input features and target labels over time (e.g., changes in customer shopping habits, economic conditions, or competition).
- Deployed models must be monitored and periodically retrained to counter this effect.

---

### Question 6
**Answer:** VotingClassifier (≈0.85)

**Explanation:**
- The exact test set ROC-AUC scores calculated from the MP5 checkpoint are:
  - **Voting Classifier:** **0.8508**
  - **Gradient Boosting:** **0.8495**
  - **Random Forest:** **0.8457**
  - **Logistic Regression:** **0.8352**
- Thus, the Voting Classifier achieves the highest overall ROC-AUC.

---

### Question 7
**Answer:** ROC-AUC measures discrimination across all thresholds, but profit depends on predictions at one specific threshold — a model that is well-calibrated in the high-precision region can outperform one with better overall AUC.

**Explanation:**
- ROC-AUC measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. It integrates performance over all possible threshold values.
- Business campaigns operate at a single specific threshold. A model with a lower overall AUC but better calibration and precision in the specific region of the optimal threshold will yield higher profits.

---

### Question 8
**Answer:** Consider business profit at the optimal threshold, model interpretability, fairness across demographic groups, and deployment complexity.

**Explanation:**
- Statistical metrics like ROC-AUC are only one dimension of model selection. 
- Real-world deployment decisions must balance financial utility (profit), legal/regulatory compliance (fairness), explainability (interpretability), and operational maintenance overhead (complexity).

---

### Question 9
**Answer:** The model treats both gender groups approximately equally in terms of identifying at-risk customers.

**Explanation:**
- **Calculation:**
  - Male Recall: $0.4078$
  - Female (K) Recall: $0.4239$
  - Recall Gap: $|0.4078 - 0.4239| = 0.0161$ (1.61%)
- Since the gap is well below the 5% threshold ($0.05$), the model demonstrates no significant gender bias.

---

### Question 10
**Answer:** “LogisticRegression produces the highest campaign profit, its coefficients directly show which factors drive lapse, and it treats gender groups fairly — making it the safest choice to explain to the board and to regulators”

**Explanation:**
- **Profit:** Logistic Regression yields the highest overall profit of **440.0 PLN** at its optimal threshold of 0.45.
- **Explainability:** Logistic Regression is transparent and simple to explain using its linear coefficients.
- **Fairness:** At the optimal threshold (0.45), the recall gap is just $0.0370$ (3.7%), which is safe and fair.
- Therefore, Logistic Regression is the most robust and defensible recommendation for the business.
