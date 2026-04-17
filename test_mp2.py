import pandas as pd
import numpy as np

# Check Q3 "Not applicable" vs "No" pd.get_dummies
df_cat = pd.DataFrame({"col": ["Yes", "No", "Not applicable"]})
print(pd.get_dummies(df_cat, drop_first=True))
# Wait, "No" < "Not applicable" ?
# len("No") = 2, "No"[0]='N', "No"[1]='o'
# len("Not") = 3, "Not"[0]='N', "Not"[1]='o', "Not"[2]='t'
# wait! 'No ' vs 'Not '
# "No" comes before "Not"
l = sorted(["Yes", "No", "Not applicable"])
print("sorted: ", l)

# Now, implement mp2 preprocessing
DATA_DIR = "/home/mp/PUM_ENG_Project_2026/2. data"
customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
y = customers["is_lapsed"].copy()
gender_series = customers["gender"].copy()
df = customers.drop(columns=["customer_id", "is_lapsed"])

# Step 4
POLISH_MONTHS = {
    "sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6,
    "lip": 7, "sie": 8, "wrz": 9, "paź": 10, "lis": 11, "gru": 12,
}
def parse_polish_date(date_str):
    day, month_str, year = date_str.split("-")
    return pd.Timestamp(year=int(year), month=POLISH_MONTHS[month_str], day=int(day))
df["registration_date"] = df["registration_date"].apply(parse_polish_date)
df = df.drop(columns=["registration_date"])

# Step 5
df["total_spend"] = df["total_spend"].str.replace("PLN ", "").str.replace(",", "").astype(float)
print("Mean total_spend:", df["total_spend"].mean())
print("Median total_spend:", df["total_spend"].median())

# Step 6
imp = (df["satisfaction_score"] < 1.0) | (df["satisfaction_score"] > 5.0)
print("Impossible scores count:", imp.sum())
df.loc[imp, "satisfaction_score"] = np.nan

# Step 7
for col in ["age", "online_ratio", "satisfaction_score"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["monthly_income_bracket", "referral_source"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["age"] = df["age"].astype(int)

# Step 8
Q1 = df["avg_basket_value"].quantile(0.25)
Q3 = df["avg_basket_value"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (df["avg_basket_value"] >= lower_bound) & (df["avg_basket_value"] <= upper_bound)
print("Rows removed:", (~mask).sum())
df = df[mask]
y = y[mask]
gender_series = gender_series[mask]

# Step 9
df["loyalty_member"] = df["loyalty_member"].map({"Tak": 1, "Nie": 0})
df["monthly_income_bracket"] = df["monthly_income_bracket"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})
df = pd.get_dummies(df, drop_first=False)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
df = df[sorted(df.columns)]
print("Shape after encoding:", df.shape)
print("Number of columns after drop_first=False:", len(df.columns))

# Step 10 & 11
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
