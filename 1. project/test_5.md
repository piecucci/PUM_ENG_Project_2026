# MP5: Model Comparison & Final Recommendation - Test 5

This document contains the questions for Test 5 based on the MP5 requirements.

---

## Question 1
A data scientist argues: “GradientBoosting has the highest ROC-AUC, so we should deploy it.” What is the strongest counterargument?
- ROC-AUC is not a valid metric for imbalanced datasets
- GradientBoosting models cannot be deployed to production
- The model with the best statistical metric may not produce the highest business profit — threshold optimization and cost matrix analysis should also be considered
- ROC-AUC only matters if the dataset has more than 10,000 samples

## Question 2
When comparing models for fairness, Model A has recall 0.40 (M) and 0.38 (K), while Model B has 0.60 (M) and 0.25 (K). Which raises a bigger fairness concern?
- Model A, because both recall values are below 0.5
- Model B, because the 0.35 gap in recall means the model is much better at identifying at-risk customers in one gender group
- Neither — fairness only applies to groups with equal representation
- Model A, because it discriminates equally poorly against both groups

## Question 3
Rank from MOST to LEAST interpretable: (1) Logistic Regression, (2) Random Forest, (3) VotingClassifier (LR + RF + GB).
- 3, 2, 1
- 1, 2, 3
- 2, 1, 3
- All three are equally interpretable through SHAP values

## Question 4
A VotingClassifier with voting="soft" combines predictions from LR, RF, and GB. How does it determine the final prediction?
- Each model casts a binary vote (0 or 1), and the majority wins
- The class probabilities from each model are averaged, and the class with the highest average probability is predicted
- The model with the highest individual accuracy's prediction is always used
- The predictions are weighted by each model's ROC-AUC score

## Question 5
A model was trained on 2022–2024 data. What is the biggest risk of deploying it without updates?
- The ROC-AUC will decrease by exactly 0.01 per month
- The model will crash when it encounters new customer IDs
- Concept drift — customer behavior may change over time, causing the model's assumptions to become invalid
- The model will overfit to the test set after deployment

## Question 6
Which model achieves the highest ROC-AUC on the test set in MP5?
- LogisticRegression (≈0.84)
- RandomForest (≈0.85)
- GradientBoosting (≈0.85)
- VotingClassifier (≈0.85)

## Question 7
In your multi-model comparison, the model with the highest ROC-AUC does not produce the highest business profit. What is the most likely explanation?
- ROC-AUC measures discrimination across all thresholds, but profit depends on predictions at one specific threshold — a model that is well-calibrated in the high-precision region can outperform one with better overall AUC
- The profit calculation has a bug — a model with higher AUC should always produce higher profit
- The cost matrix is wrong — campaign costs should not be subtracted from revenue
- This indicates overfitting — the model with higher AUC memorized the training data

## Question 8
GradientBoosting achieves ROC-AUC ≈ 0.85, compared to RandomForest’s ≈ 0.85 and LogisticRegression’s ≈ 0.84. Given these near-identical AUCs, what additional criteria should guide model selection?
- Always choose the model with the highest AUC, even if the difference is 0.01
- Consider business profit at the optimal threshold, model interpretability, fairness across demographic groups, and deployment complexity
- Choose the simplest model (LR) because Occam’s razor always applies in machine learning
- Choose the ensemble (VotingClassifier) because combining models always improves performance

## Question 9
In the fairness analysis, the recall gap between genders M and K for GradientBoosting is less than 0.02 (M: 0.408, K: 0.424). What does this indicate?
- The model is biased against male customers
- The model is biased against female customers
- The model treats both gender groups approximately equally in terms of identifying at-risk customers
- The fairness analysis is invalid because gender was used as a feature

## Question 10
You must recommend one model to MajsterPlus’s VP of Marketing. She asks: “Why should I trust this model?” Which argument is most appropriate?
- “GradientBoosting has the highest AUC among individual models (0.85), and higher AUC always means better business outcomes”
- “RandomForest shows which features matter most through feature importance scores, making it easy to explain predictions to the board”
- “VotingClassifier combines three models for the most robust predictions, and its AUC is the highest overall”
- “LogisticRegression produces the highest campaign profit, its coefficients directly show which factors drive lapse, and it treats gender groups fairly — making it the safest choice to explain to the board and to regulators”
