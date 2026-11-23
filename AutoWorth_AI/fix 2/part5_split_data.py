# =====================================================================
# PART 5: Split the Data  (70% Train / 15% Validation / 15% Test)
# =====================================================================
# Input : combined_cars_clean.csv   (output of Part 3)
# Output: train.csv, val.csv, test.csv, 09_split_price_distribution.png
#
# Ground rules followed in this script:
#   1. Split FIRST. Nothing is imputed, encoded or scaled here, and no
#      statistic is computed from the full dataset for later use.
#      (The engineSize NaNs stay NaN -- the imputer is fit on Train only, in Part 6.)
#   2. One plain random split: shuffle=True, random_state=42.
#      No stratify= (price is continuous), and no re-seeding until the split
#      "looks nicer" -- that would be tuning the split on the test set.
#   3. Every check below is a sanity check only; none of them changes a split.
#   4. test.csv is written to disk but not used again until the final model is chosen.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "price"
CAT_COLS = ["model", "transmission", "fuelType", "Make"]

# ----------------------------------------------------------------------
# 5.0 Load + pre-split checkpoint
# ----------------------------------------------------------------------
df = pd.read_csv("combined_cars_clean.csv")

print("=" * 70)
print("5.0 LOAD + PRE-SPLIT CHECKPOINT")
print("=" * 70)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Missing values:", missing.to_dict() if len(missing) else "none")
assert set(missing.index) <= {"engineSize"}, "Unexpected missing values -- re-check Part 3 output"
assert df.duplicated().sum() == 0, "Duplicate rows found -- re-check Part 3 output"

# Part 3's final decisions (remove year < 1980 rows and 'A Class' rows with engineSize > 2.1L)
# must already be applied BEFORE splitting, so Train/Val/Test all come from the same final
# dataset. These three counts should all be 0 -- if not, re-run Part 3 first.
is_a_class = df["model"].str.strip() == "A Class"
n_a_class = (is_a_class & (df["price"] > 100000)).sum()
n_a_class_engine = (is_a_class & (df["engineSize"] > 2.1)).sum()
n_old = (df["year"] < 1980).sum()
print(f"Part 3 items that should be gone: 'A Class' > £100k = {n_a_class}, "
      f"'A Class' engineSize > 2.1L = {n_a_class_engine}, year < 1980 = {n_old}")
if n_a_class or n_a_class_engine or n_old:
    print("-> WARNING: re-run Part 3 (final version), then re-run this script.")
print()

# ----------------------------------------------------------------------
# 5.1 Split: 70 / 15 / 15
# ----------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Step 1: 70% train / 30% temporary hold-out
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, shuffle=True
)
# Step 2: split the 30% in half -> 15% validation / 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, shuffle=True
)

train_df = df.loc[X_train.index]
val_df = df.loc[X_val.index]
test_df = df.loc[X_test.index]
splits = {"Train": train_df, "Validation": val_df, "Test": test_df}

# ----------------------------------------------------------------------
# 5.2 Verify sizes and integrity
# ----------------------------------------------------------------------
n = len(df)
size_table = pd.DataFrame({"rows": {k: len(v) for k, v in splits.items()}})
size_table["percent"] = (size_table["rows"] / n * 100).round(2)

print("=" * 70)
print("5.2 SPLIT SIZES")
print("=" * 70)
print(size_table)
print(f"Total: {size_table['rows'].sum()}  (dataset: {n})")

assert size_table["rows"].sum() == n, "Rows lost or duplicated by the split"
idx = {k: set(v.index) for k, v in splits.items()}
assert not (idx["Train"] & idx["Validation"]), "Train/Validation overlap"
assert not (idx["Train"] & idx["Test"]), "Train/Test overlap"
assert not (idx["Validation"] & idx["Test"]), "Validation/Test overlap"
assert len(idx["Train"] | idx["Validation"] | idx["Test"]) == n, "Some rows are in no split"
assert pd.concat(list(splits.values())).duplicated().sum() == 0, "Identical rows across splits"
print("OK: no overlap between splits, every row used exactly once, no identical rows across splits.")
print()

# ----------------------------------------------------------------------
# 5.3 Target (price) distribution across splits
# ----------------------------------------------------------------------
dist = pd.DataFrame({k: v[TARGET].describe() for k, v in splits.items()}).T
dist["skew"] = [v[TARGET].skew() for v in splits.values()]

print("=" * 70)
print("5.3 PRICE DISTRIBUTION BY SPLIT")
print("=" * 70)
print(dist.round(2))
print()
for name in ["Validation", "Test"]:
    stat, p = ks_2samp(train_df[TARGET], splits[name][TARGET])
    print(f"KS test, Train vs {name}: statistic = {stat:.4f}, p-value = {p:.3f}")
print("(KS statistic = largest gap between the two price CDFs; closer to 0 = more similar.")
print(" Judge by the statistic, not the p-value: with ~14k rows even trivial gaps can give p < 0.05,")
print(" and a purely random split does so about 1 time in 20 by chance.")
print(" Do NOT change the seed to chase a nicer number.)")
print()

plt.figure(figsize=(9, 5))
bins = np.linspace(np.log1p(df[TARGET].min()), np.log1p(df[TARGET].max()), 60)
for name, part in splits.items():
    plt.hist(np.log1p(part[TARGET]), bins=bins, density=True, histtype="step",
             linewidth=2, label=f"{name} (n={len(part)})")
plt.title("Log Price Distribution by Split (density-normalised)")
plt.xlabel("log(1 + Price)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("09_split_price_distribution.png")
plt.show()
plt.close()

# ----------------------------------------------------------------------
# 5.4 Categorical balance + categories unseen in Train
# ----------------------------------------------------------------------
print("=" * 70)
print("5.4 CATEGORICAL BALANCE")
print("=" * 70)
make_pct = pd.DataFrame(
    {k: v["Make"].value_counts(normalize=True) * 100 for k, v in splits.items()}
).round(2)
print("Make share (%) per split:")
print(make_pct)
print(f"Max |Validation - Train| gap: {(make_pct['Validation'] - make_pct['Train']).abs().max():.2f} pp")
print(f"Max |Test - Train| gap:       {(make_pct['Test'] - make_pct['Train']).abs().max():.2f} pp")
print()

print("Categories present in Validation/Test but never seen in Train:")
found_unseen = False
for col in CAT_COLS:
    train_vals = set(train_df[col])
    for name in ["Validation", "Test"]:
        part = splits[name]
        unseen = part[~part[col].isin(train_vals)]
        if len(unseen):
            found_unseen = True
            print(f"  {col:13s} {name:10s}: {len(unseen)} rows, "
                  f"{unseen[col].nunique()} categories -> {sorted(unseen[col].unique())[:10]}")
if not found_unseen:
    print("  none")
else:
    print("-> Use an encoder that tolerates unseen categories in Part 6 "
          "(e.g. OneHotEncoder(handle_unknown='ignore')). Do not move rows between splits to avoid this.")
print()

# ----------------------------------------------------------------------
# 5.5 Missing engineSize per split (NOT imputed here)
# ----------------------------------------------------------------------
print("=" * 70)
print("5.5 MISSING engineSize PER SPLIT (left as NaN -- imputer will be fit on Train only)")
print("=" * 70)
for name, part in splits.items():
    print(f"  {name:10s}: {part['engineSize'].isna().sum()} NaN")
print()

# ----------------------------------------------------------------------
# 5.6 Save
# ----------------------------------------------------------------------
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("=" * 70)
print("SPLIT LOG")
print("=" * 70)
print(f"1. Random 70/15/15 split of {n} cleaned rows (shuffle=True, random_state={SEED}), done in two steps: "
      f"70/30, then the 30% halved.")
print("2. No stratification: price is continuous. Balance was checked afterwards (5.3, 5.4), not forced.")
print("3. No preprocessing has been fitted. Imputation, encoding and scaling will be fit on Train only, "
      "then applied to Validation and Test.")
print("4. Validation is used for model comparison and tuning. Test is used once, after the final model is chosen.")
print("5. Outliers kept in Part 3 stay in all three splits, including Test -- they are not removed after the fact.")
print("6. Caveat: a random split treats rows as independent; near-identical listings (not exact duplicates) "
      "can land in different splits and make Test scores slightly optimistic.")
print()
print("Saved: train.csv, val.csv, test.csv, 09_split_price_distribution.png")
