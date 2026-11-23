# =====================================================================
# PART 7: Data Preprocessing
# =====================================================================
# Input : train_fe.csv, val_fe.csv, test_fe.csv   (output of Part 6)
# Output: preprocessor.joblib        -> the fitted ColumnTransformer (fit on Train only)
#         preprocessed_data.joblib   -> dict: X_train/X_val/X_test (DataFrames) + y_train/y_val/y_test
#         feature_names.csv          -> list of the final model-input columns
#
# القواعد:
#   1. أي حاجة بتتعلّم من الداتا (median بتاع الـ imputer، mean/std بتاع الـ scaler،
#      قايمة الـ categories بتاعة الـ encoder) بتتعمل fit على X_train بس. بعدين نفس الـ
#      fitted object بيتطبق (transform بس) على Validation و Test.
#   2. price (الـ target) مش بيدخل الـ preprocessor ومش بيتحوّل هنا -- القرار ده بتاع مرحلة الموديل.
#   3. year مش داخل الموديل: car_age = 2020 - year بالظبط (Part 6)، يعني نفس المعلومة 100%
#      وبس بتتكرر. (لو عايز تبقيها: KEEP_YEAR_COLUMN = True تحت.)
#   4. مفيش outliers بتتشال أو بتتقصّ هنا -- القرارات دي خلصت في Part 3.

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_major, _minor = (int(v) for v in sklearn.__version__.split(".")[:2])
assert (_major, _minor) >= (1, 2), "scikit-learn >= 1.2 is required (sparse_output / set_output)"

TARGET = "price"
AGE_REFERENCE_YEAR = 2020        # لازم يطابق الثابت اللي اتستخدم في Part 6
KEEP_YEAR_COLUMN = False         # year و car_age نفس المعلومة بالظبط -> الافتراضي نشيل year

CONTINUOUS = ["mileage", "tax", "mpg", "engineSize", "car_age", "mileage_per_year"]
if KEEP_YEAR_COLUMN:
    CONTINUOUS = ["year"] + CONTINUOUS
BINARY = ["is_premium_brand"]
CATEGORICAL = ["Make", "model", "transmission", "fuelType"]
FEATURES = CONTINUOUS + BINARY + CATEGORICAL

# ----------------------------------------------------------------------
# 7.0 Load + validate the Part 6 outputs
# ----------------------------------------------------------------------
raw = {
    "Train": pd.read_csv("train_fe.csv"),
    "Validation": pd.read_csv("val_fe.csv"),
    "Test": pd.read_csv("test_fe.csv"),
}

print("=" * 70)
print("7.0 LOAD + VALIDATE")
print("=" * 70)
for name, part in raw.items():
    missing_cols = (set(FEATURES) | {TARGET, "year"}) - set(part.columns)
    assert not missing_cols, f"{name}: missing columns {missing_cols} -- re-run Part 6"

    nan_counts = part.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    assert set(nan_counts.index) <= {"engineSize", "mileage_per_year"}, (
        f"{name}: unexpected NaNs {nan_counts.to_dict()}"
    )
    n_age0 = int((part["car_age"] == 0).sum())
    assert int(part["mileage_per_year"].isna().sum()) == n_age0, (
        f"{name}: mileage_per_year NaNs should equal the car_age == 0 rows"
    )
    assert ((part["year"] + part["car_age"]) == AGE_REFERENCE_YEAR).all(), (
        f"{name}: car_age != {AGE_REFERENCE_YEAR} - year for some rows -- check Part 6 constant"
    )
    print(f"  {name:10s}: {part.shape[0]:6d} rows | NaN: {nan_counts.to_dict()} | "
          f"car_age == 0 rows: {n_age0}")
print("OK: only engineSize / mileage_per_year have NaNs; mileage_per_year NaN == car_age 0 rows; "
      f"car_age == {AGE_REFERENCE_YEAR} - year holds on every row.")
print()

# ----------------------------------------------------------------------
# 7.1 Define X / y and the feature groups
# ----------------------------------------------------------------------
X = {name: part[FEATURES].copy() for name, part in raw.items()}   # price (and year) are NOT in X
y = {name: part[TARGET].copy() for name, part in raw.items()}

print("=" * 70)
print("7.1 FEATURE GROUPS")
print("=" * 70)
print(f"Target                       : {TARGET}  (never passed to the preprocessor, not transformed here)")
print(f"Continuous numerical ({len(CONTINUOUS)})    : {CONTINUOUS}")
print(f"Binary ({len(BINARY)})                    : {BINARY}  -> passed through as 0/1")
print(f"Categorical ({len(CATEGORICAL)})            : {CATEGORICAL}")
if not KEEP_YEAR_COLUMN:
    print("Excluded                     : year -- exact duplicate of car_age "
          f"(car_age = {AGE_REFERENCE_YEAR} - year, verified on every row above)")
print()

# ----------------------------------------------------------------------
# 7.2 Text cleanup (deterministic, nothing is learned)
# ----------------------------------------------------------------------
# قيم زي ' A Class' فيها مسافة في الأول. الـ strip *بيدمج* variants (مثلاً ' A Class' و 'A Class'
# بيبقوا category واحدة) -- ودي نتيجة صح ومطلوبة، لأن ده نفس الموديل. Part 3 بتطبع الـ collisions
# بس مابتصلّحهاش، فالدمج بيحصل هنا. بنعرض كل مجموعة اندمجت، ونتأكد إن مفيش مسافات طرفية فاضلة.
# لازم يتطبق على الـ 3 splits (مش Train بس) عشان قيم Val/Test اللي فيها مسافة ماتبقاش "unseen".
for name, Xp in X.items():
    for col in CATEGORICAL:
        original = Xp[col].copy()
        stripped = original.str.strip()

        pairs = pd.DataFrame({"orig": original, "clean": stripped}).drop_duplicates()
        merged = pairs.groupby("clean")["orig"].apply(list)
        merged = merged[merged.map(len) > 1]
        if len(merged):
            print(f"  {name}.{col}: strip() merged {len(merged)} group(s):")
            for clean, variants in merged.items():
                counts = [int((original == v).sum()) for v in variants]
                print(f"      {variants} -> {clean!r}  (rows: {counts})")

        Xp[col] = stripped
        assert (Xp[col] == Xp[col].str.strip()).all(), f"{name}.{col}: leading/trailing spaces remain"
print("7.2 Text cleanup: str.strip() applied to categorical columns in every split "
      "(variants that differed only by surrounding spaces were merged; see list above).")
print()

# ----------------------------------------------------------------------
# 7.3 Build the preprocessor (nothing is fitted yet)
# ----------------------------------------------------------------------
# continuous : median imputation (NaN في engineSize و mileage_per_year) ثم StandardScaler
#              -- الـ scaling ملوش أي تأثير على تنبؤات الـ tree models (اتأكدنا بتجربة)،
#              وبيفيد Ridge/Lasso/KNN/SVR، فـ preprocessor واحد يخدم كل الموديلات.
# binary     : passthrough
# categorical: OneHotEncoder(handle_unknown='ignore'):
#              * أي category ظهرت في Val/Test ومشافهاش الـ Train بتتحوّل لـ all-zeros بدل ما يحصل error.
#              * من غير drop='first': مع drop='first' الـ unseen بيبقى شكله زي الـ category المحذوفة
#                بالظبط (all-zeros برضه) فبيتلخبطوا.
preprocessor = ColumnTransformer(
    transformers=[
        ("cont", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())]), CONTINUOUS),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ],
    remainder="drop",
)
preprocessor.set_output(transform="pandas")

# ----------------------------------------------------------------------
# 7.4 FIT on Train only -> TRANSFORM Train / Validation / Test
# ----------------------------------------------------------------------
preprocessor.fit(X["Train"])                      # <- the ONLY fit call. No y is passed.
P = {name: preprocessor.transform(Xp) for name, Xp in X.items()}

X_train, X_val, X_test = X["Train"], X["Validation"], X["Test"]
X_train_p, X_val_p, X_test_p = P["Train"], P["Validation"], P["Test"]
y_train, y_val, y_test = y["Train"], y["Validation"], y["Test"]

print("=" * 70)
print("7.4 FIT ON TRAIN ONLY, TRANSFORM ALL THREE")
print("=" * 70)
for name in P:
    print(f"  {name:10s}: {X[name].shape[1]} raw features -> {P[name].shape[1]} processed columns, "
          f"{P[name].shape[0]} rows")
print()

# ----------------------------------------------------------------------
# 7.5 Verification
# ----------------------------------------------------------------------
print("=" * 70)
print("7.5 VERIFICATION")
print("=" * 70)

# (a) الـ learned statistics لازم تطابق إحصائيات الـ Train اللي اتحسبت بشكل مستقل
imputer = preprocessor.named_transformers_["cont"].named_steps["imputer"]
scaler = preprocessor.named_transformers_["cont"].named_steps["scaler"]
print("(a) Learned continuous-feature statistics == independently computed Train statistics:")
for i, col in enumerate(CONTINUOUS):
    med = X_train[col].median()
    filled = X_train[col].fillna(med)
    assert np.isclose(imputer.statistics_[i], med)
    assert np.isclose(scaler.mean_[i], filled.mean())
    assert np.isclose(scaler.scale_[i], filled.std(ddof=0))
    print(f"    {col:18s} median={imputer.statistics_[i]:12.3f}  mean={scaler.mean_[i]:12.3f}  "
          f"std={scaler.scale_[i]:12.3f}")
print("    OK -- imputer median and scaler mean/std come from Train only.")

# (b) الـ encoder شاف Train بس
ohe = preprocessor.named_transformers_["cat"]
print("\n(b) OneHotEncoder categories == Train categories (Validation/Test-only categories NOT learned):")
n_unseen = {}
for j, col in enumerate(CATEGORICAL):
    train_cats = set(X_train[col])
    assert set(ohe.categories_[j]) == train_cats
    for name in ["Validation", "Test"]:
        n_unseen[(col, name)] = int((~X[name][col].isin(train_cats)).sum())
    print(f"    {col:13s}: {len(train_cats):4d} categories learned from Train")
print("    OK.")

# (c) أي category مش في Train لازم تطلع all-zeros، والمعروفة لازم يبقى فيها 1 واحد بالظبط
print("\n(c) Unseen categories are encoded as all-zeros (no error); known categories get exactly one 1:")
for col in CATEGORICAL:
    block = [c for c in X_train_p.columns if c.startswith(f"cat__{col}_")]
    assert len(block) == len(ohe.categories_[CATEGORICAL.index(col)])
    train_cats = set(X_train[col])
    for name in ["Validation", "Test"]:
        known = X[name][col].isin(train_cats)
        block_sum = P[name][block].sum(axis=1)
        assert (block_sum[known] == 1).all()
        assert (block_sum[~known] == 0).all()
        if n_unseen[(col, name)]:
            print(f"    {col:13s} {name:10s}: {n_unseen[(col, name)]} rows with a category unseen in Train -> all-zeros")
print("    OK.")

# (d) Train متوسّطه 0 وانحرافه 1 بالتصميم؛ Validation/Test لأ -- ودي علامة إن الـ scaler ما شافهمش
cont_cols = [f"cont__{c}" for c in CONTINUOUS]
print("\n(d) Scaled continuous columns -- Train is ~0/1 by construction; Validation/Test are NOT exactly 0/1 "
      "(expected, proves they were not used for fitting):")
for name in P:
    means = P[name][cont_cols].mean()
    stds = P[name][cont_cols].std(ddof=0)
    print(f"    {name:10s}: max |mean| = {means.abs().max():.4f} | std range = [{stds.min():.3f}, {stds.max():.3f}]")
assert np.allclose(X_train_p[cont_cols].mean(), 0, atol=1e-8)
assert np.allclose(X_train_p[cont_cols].std(ddof=0), 1, atol=1e-8)

# (e) sanity عام
print("\n(e) General checks:")
for name in P:
    assert not P[name].isnull().any().any(), f"{name}: NaN left after preprocessing"
    assert P[name].index.equals(X[name].index), f"{name}: row order/index changed"
    assert list(P[name].columns) == list(X_train_p.columns), f"{name}: columns differ from Train"
    assert TARGET not in P[name].columns
if not KEEP_YEAR_COLUMN:
    assert "cont__year" not in X_train_p.columns
assert not any(TARGET in c for c in X_train_p.columns)
print("    No NaN left | same columns in all splits | row order preserved | price not among the features.")
print()

# ----------------------------------------------------------------------
# 7.6 Save
# ----------------------------------------------------------------------
feature_table = pd.DataFrame({
    "feature": X_train_p.columns,
    "group": [c.split("__")[0] for c in X_train_p.columns],
})
feature_table.to_csv("feature_names.csv", index=False)
joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(
    {"X_train": X_train_p, "X_val": X_val_p, "X_test": X_test_p,
     "y_train": y_train, "y_val": y_val, "y_test": y_test},
    "preprocessed_data.joblib", compress=3,
)

group_counts = feature_table["group"].value_counts().to_dict()
print("=" * 70)
print("PREPROCESSING LOG")
print("=" * 70)
print(f"1. Target: {TARGET}. Features: {len(CONTINUOUS)} continuous + {len(BINARY)} binary + "
      f"{len(CATEGORICAL)} categorical -> {X_train_p.shape[1]} columns after encoding {group_counts}.")
print("2. Continuous: median imputation (engineSize NaNs; mileage_per_year NaNs = car_age 0 rows) "
      "then StandardScaler. Both fitted on Train only.")
print("3. mileage_per_year NaN is 'undefined' (division by zero), not random missingness. No extra "
      "'is_new_car' flag is added: car_age == 0 already carries that information.")
print("4. Binary is_premium_brand passed through unchanged.")
print("5. Categorical: OneHotEncoder(handle_unknown='ignore'), no drop; categories learned from Train only. "
      "Unseen Validation/Test categories become all-zeros.")
print(f"6. {'year kept' if KEEP_YEAR_COLUMN else 'year excluded'}: car_age = {AGE_REFERENCE_YEAR} - year exactly, "
      "so the two columns hold identical information.")
print("7. The fitted preprocessor was applied unchanged to Validation and Test. Nothing was fit on them.")
print("8. price is not transformed here. If a log-target is tried later, it is a model-stage decision judged "
      "on Validation, and predictions must be inverted (expm1) before computing errors in £.")
print()
print("Saved: preprocessor.joblib, preprocessed_data.joblib, feature_names.csv")
