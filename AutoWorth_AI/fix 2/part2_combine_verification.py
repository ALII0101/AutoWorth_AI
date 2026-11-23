# =====================================================================
# PART 2: Post-Combine Verification
# =====================================================================
# ده بيكمل على نفس متغيرات الجزء الأول في نفس الـ Colab session:
#   raw_dataframes  -> dict: Make -> DataFrame خام (قبل الدمج)
#   combined_cars   -> DataFrame المدموج (بعد rename + concat)
#   REFERENCE_YEAR  -> int ثابت (معرّف في أول خلية بالنوتبوك)، بيحل محل
#                      datetime.now().year عشان الفحوصات تفضل reproducible
# لو الـ session اتقفلت، لازم تعيد تشغيل كود الجزء الأول الأول.

import pandas as pd

# =====================================================================
# 1. Row Count Check — تأكيد إن الـ concat ما فقدش/كررش صفوف
# =====================================================================
expected_rows = sum(df.shape[0] for df in raw_dataframes.values())
assert len(combined_cars) == expected_rows, (
    f"Row count mismatch! expected {expected_rows}, got {len(combined_cars)}"
)
print(f"Row count OK: {len(combined_cars)} rows (expected {expected_rows})")

# =====================================================================
# 2. Missing Values (recheck after combining)
# =====================================================================
missing = combined_cars.isnull().sum()
missing_cols = missing[missing > 0]
print("\nMissing values after combining:")
print("None found." if missing_cols.empty else missing_cols)

# =====================================================================
# 3. Duplicates (recheck after combining) — تسجيل بس، من غير حذف
# =====================================================================
dup_count = combined_cars.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}  (kept as-is — decision belongs to the cleaning stage)")

# =====================================================================
# 4. Data Types (recheck, split numeric vs categorical)
# =====================================================================
numeric_cols = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
categorical_cols = ["model", "transmission", "fuelType", "Make"]

print("\nNumeric dtypes:")
print(combined_cars[numeric_cols].dtypes)
print("\nCategorical dtypes:")
print(combined_cars[categorical_cols].dtypes)
# mpg و engineSize فيهم قيم عشرية فعلاً -> float64 صح، مفيش داعي لأي astype() هنا.

# =====================================================================
# 5. Suspicious / Unrealistic Values — Detect & Inspect only, NO deleting
# =====================================================================
# REFERENCE_YEAR بدل datetime.now().year: عشان لو حد شغّل الكود بعدين،
# حد الـ"invalid future year" يفضل ثابت زي ما وثقناه، مش بيتغير مع كل تشغيل.
current_year = REFERENCE_YEAR

checks = {
    f"year > {current_year}": combined_cars["year"] > current_year,
    "engineSize <= 0":        combined_cars["engineSize"] <= 0,
    "mpg > 200":               combined_cars["mpg"] > 200,
    "price > 100000":         combined_cars["price"] > 100000,
    "price < 500":            combined_cars["price"] < 500,   # إضافة: أقل سعر عندنا 450
    "mileage > 200000":       combined_cars["mileage"] > 200000,
}

print("\nSuspicious value counts:")
for label, mask in checks.items():
    print(f"  {label}: {mask.sum()}")

# ---- Year outliers ----
year_outliers = combined_cars[combined_cars["year"] > current_year]
print(f"\n[year > {current_year}] {len(year_outliers)} rows:")
print(year_outliers[["Make", "model", "year", "price"]]
      .sort_values("year", ascending=False).head(15))

# ---- engineSize <= 0 — لازم نتأكد إنها مش عربيات Electric طبيعي مالهاش محرك ----
zero_engine = combined_cars[combined_cars["engineSize"] <= 0]
print(f"\n[engineSize <= 0] {len(zero_engine)} rows — breakdown by fuelType:")
print(zero_engine["fuelType"].value_counts())
non_electric_zero = zero_engine[zero_engine["fuelType"] != "Electric"]
print(f"  -> {len(non_electric_zero)} of these are NOT Electric (the real suspects)")

# ---- mpg > 200 — عادة بتظهر في Hybrid/PHEV بسبب طريقة الحساب الرسمية في UK ----
high_mpg = combined_cars[combined_cars["mpg"] > 200]
print(f"\n[mpg > 200] {len(high_mpg)} rows — breakdown by fuelType:")
print(high_mpg["fuelType"].value_counts())

# ---- Price outliers (high & low) ----
high_price = combined_cars[combined_cars["price"] > 100000]
print(f"\n[price > 100000] {len(high_price)} rows:")
print(high_price[["Make", "model", "year", "price"]]
      .sort_values("price", ascending=False).head(10))

low_price = combined_cars[combined_cars["price"] < 500]
print(f"\n[price < 500] {len(low_price)} rows:")
print(low_price[["Make", "model", "year", "price", "mileage"]]
      .sort_values("price").head(10))

# ---- Mileage outliers ----
high_mileage = combined_cars[combined_cars["mileage"] > 200000]
print(f"\n[mileage > 200000] {len(high_mileage)} rows:")
print(high_mileage[["Make", "model", "year", "mileage", "price"]]
      .sort_values("mileage", ascending=False).head(10))

print("\nDone. النتائج دي استكشاف بس — قرار الحذف/التصحيح يتأجل لمرحلة Data Cleaning.")
