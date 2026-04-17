import json
import re

with open("3. notebooks/mp2_starter.ipynb", "r") as f:
    nb = json.load(f)

replacements = [
    (
        "# TODO: Apply the parse_polish_date helper to the registration_date column.",
        """# Apply the parse_polish_date helper
df["registration_date"] = df["registration_date"].apply(parse_polish_date)
df = df.drop(columns=["registration_date"])"""
    ),
    (
        "# TODO: Convert total_spend from currency string format",
        """df["total_spend"] = df["total_spend"].str.replace("PLN ", "").str.replace(",", "").astype(float)
print(df["total_spend"].describe())"""
    ),
    (
        "# TODO: Print the mean and median of total_spend explicitly.",
        """print(f"Mean: {df['total_spend'].mean()}")
print(f"Median: {df['total_spend'].median()}")
# Distribution is right-skewed"""
    ),
    (
        "# TODO: Identify satisfaction_score values outside the valid range",
        """imp = (df["satisfaction_score"] < 1.0) | (df["satisfaction_score"] > 5.0)
print("Impossible scores found:", imp.sum())
df.loc[imp, "satisfaction_score"] = float('nan')"""
    ),
    (
        "# TODO: Impute all missing values.",
        """for col in ["age", "online_ratio", "satisfaction_score"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["monthly_income_bracket", "referral_source"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["age"] = df["age"].astype(int)
print(df.isnull().sum())"""
    ),
    (
        "# TODO: Filter out rows where avg_basket_value",
        """mask = (df["avg_basket_value"] >= lower_bound) & (df["avg_basket_value"] <= upper_bound)
print("Rows removed:", (~mask).sum())
df = df[mask]
y = y[mask]
gender_series = gender_series[mask]"""
    ),
    (
        "# TODO: Encode all categorical variables following the order above.",
        """df["loyalty_member"] = df["loyalty_member"].map({"Tak": 1, "Nie": 0})
df["monthly_income_bracket"] = df["monthly_income_bracket"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})

df = pd.get_dummies(df, drop_first=False)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
df = df[sorted(df.columns)]
print(df.shape)
print(df.dtypes)"""
    ),
    (
        "# TODO: Write your answer as a comment or print statement.\n# Consider: What is multicollinearity?",
        """# `drop_first=True` avoids perfect multicollinearity,
# which is needed for linear models. For tree-based models, drop_first=False is fine."""
    ),
    (
        "# TODO: Apply StandardScaler. Fit on training data ONLY",
        """from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)"""
    ),
    (
        "# YOUR CODE HERE (optional bonus)",
        """from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
behavioral_cols = ["purchase_count", "avg_basket_value", "days_since_last_purchase", "product_categories_bought"]
kmeans.fit(X_train_scaled[behavioral_cols])
X_train_scaled["cluster"] = kmeans.labels_
X_test_scaled["cluster"] = kmeans.predict(X_test_scaled[behavioral_cols])"""
    )
]

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        for k, v in replacements:
            if k in src:
                cell["source"] = [line + "\n" for line in v.split("\n")]
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("3. notebooks/mp2_starter.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
