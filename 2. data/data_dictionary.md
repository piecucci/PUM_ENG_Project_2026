# MajsterPlus Data Dictionary

**Dataset version**: 1.0

---

## customers.csv

**Rows**: 5,000 | **Columns**: 21 (20 features + 1 target)

| # | Column | Type | Values | Description |
|---|--------|------|--------|-------------|
| 1 | `customer_id` | str | MP00...| Unique ID; join key |
| 2 | `registration_date`| str | sty... | Registration date (PL months) |
| 3 | `age` | int | 18–78  | Age in years |
| 4 | `gender` | cat | M, K   | Gender (K = Kobieta) |
| 5 | `city` | cat | 12 val | City of residence |
| 6 | `loyalty_member` | cat | Tak, Nie| Membership status |
| 7 | `total_spend` | str | PLN... | Cumulative spend (formatted) |
| 8 | `purchase_count` | int | 1–87   | Total transactions |
| 9 | `avg_basket_value` | float| 35–890 | Average transaction value |
| 10| `days_since_last` | int | 0–540  | Days since last purchase |
| 11| `categories_bought`| int | 1–8    | Number of distinct categories |
| 12| `online_ratio` | float| 0.0–1.0| Proportion of online spend |
| 13| `satisfaction` | float| 1.0–5.0| Satisfaction rating |
| 14| `service_contacts` | int | 0–15   | Customer service contacts |
| 15| `newsletter` | cat | Tak... | Subscription status |
| 16| `income_bracket` | cat | A–E    | Income bracket (A=low, E=high)|
| 17| `district_type` | cat | urban..| Type of residential area |
| 18| `store_dist_km` | float| 0.5–65 | Distance to nearest store |
| 19| `referral_source` | cat | friend..| Acquisition source |
| 20| `account_age_days` | int | ~16–1096| Days since registration |
| 21| `is_lapsed` | int | 0 / 1  | **Target** (1=lapsed, 0=act) |

---

## transactions.csv

**Rows**: ~25,000 | **Columns**: 8 | **Join key**: `customer_id`

| # | Column | Type | Values | Description |
|---|--------|------|--------|-------------|
| 1 | `transaction_id` | str | T00... | Unique identifier |
| 2 | `customer_id` | str | MP0... | Join key |
| 3 | `transaction_date`| str | PL fmt | Date of transaction |
| 4 | `store_id` | str | S01-S12| Store identifier |
| 5 | `product_category`| cat | Tools..| Category purchased |
| 6 | `amount` | float| > 0    | Transaction value (PLN) |
| 7 | `items_count` | int | 1–20   | Number of items |
| 8 | `payment_method` | cat | cash.. | Payment method |

---

## Store Mapping

| Store ID | City | Store ID | City |
|----------|------|----------|------|
| S01 | Warszawa | S07 | Szczecin |
| S02 | Krakow   | S08 | Bydgoszcz|
| S03 | Lodz     | S09 | Lublin   |
| S04 | Wroclaw  | S10 | Bialystok|
| S05 | Poznan   | S11 | Katowice |
| S06 | Gdansk   | S12 | Gdynia   |

---

## Known Data Quality Issues

| # | Issue | Column(s) |
|---|-------|-----------|
| 1 | Polish date format (sty, lut...) | `reg_date`, `trans_date` |
| 2 | Currency stored as string | `total_spend` |
| 3 | Polish categorical ("Tak"/"Nie") | `loyalty`, `newsletter` |
| 4 | Missing values (~3%) | `age` |
| 5 | Missing values (~8%) | `online_ratio` |
| 6 | Missing values (~12%) | `income_bracket` |
| 7 | Extreme outliers | `avg_basket_value` |
| 8 | Values outside 1.0–5.0 range | `satisfaction_score` |
| 9 | Missing values (~6%) | `satisfaction_score` |
| 10| Missing values (~4%) | `referral_source` |
