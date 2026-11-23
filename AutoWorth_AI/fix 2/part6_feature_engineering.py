# =====================================================================
# PART 6: Feature Engineering
# =====================================================================
# بيشتغل على الـ3 ملفات اللي خرجت من Part 5 (train.csv, val.csv, test.csv).
# الـ3 Features المختارة هنا كلهم deterministic transforms بدون أي "fitting"
# من بيانات الـTrain (مفيش mean/median/threshold متعلّم) -- فمفيش خطر Data
# Leakage بالتصميم نفسه، مش بعد ما نتأكد لاحقًا. بالتحديد بديل عن اقتراح
# "High Mileage Indicator" (اللي كان محتاج median من الـTrain بس) اخترنا
# Feature تالت مبني على معرفة خارجية ثابتة بدل إحصائية متعلّمة من الداتا.

import numpy as np
import pandas as pd

# =====================================================================
# 6.0 Load the three splits from Part 5
# =====================================================================
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

print("=" * 70)
print("6.0 LOADED SPLITS")
print("=" * 70)
for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"  {name:10s}: {part.shape[0]} rows x {part.shape[1]} columns")
print()

# =====================================================================
# 6.1 Car Age
# =====================================================================
# REFERENCE_YEAR (= 2026, معرّف في Part 1) معناه مختلف تمامًا عن المطلوب
# هنا -- هو "السنة اللي المشروع بيتنفذ فيها"، واستخدمناه بس لاكتشاف سنوات
# مستقبلية مستحيلة (year > 2026) في Part 2/3. الهدف هنا مختلف: نمثّل عمر
# العربية *وقت ما سعرها اتسجل في الداتاست*، مش وقت تشغيل الكود. آخر موديل
# year موجود فعليًا في الداتا المنضفة هو 2020 (شوف Part 4، قسم 4.4) --
# فهو أقرب تمثيل لـ"دلوقتي" بالنسبة لسوق السيارات اللي الداتا دي بتمثله،
# فعرّفنا ثابت منفصل AGE_REFERENCE_YEAR مخصوص للـFeature Engineering بس،
# بدل ما نعيد استخدام REFERENCE_YEAR (2026) لغرض تاني خالص.
#
# ملحوظة مهمة (ليه الاختيار ده مش حساس): CarAge = C - year هو مجرد تحويل
# خطي (affine) لعمود year. أي موديل خطي أو شجري (Linear/Ridge/RandomForest/
# XGBoost) هياخد بالظبط نفس القدرة التنبؤية أيًا كان C، طالما C ثابت واتطبق
# على كل صف وكل split بنفس القيمة. الفايدة الوحيدة من اختيار 2020 تحديدًا
# (بدل 2026 مثلًا) هي إن الأرقام تبقى مفهومة بشريًا (عربية موديل 2020
# تطلع عمرها ~0 مش ~6)، مش تحسين حقيقي لدقة أي موديل.
AGE_REFERENCE_YEAR = 2020


def add_car_age(df):
    df = df.copy()
    df["car_age"] = AGE_REFERENCE_YEAR - df["year"]
    return df


train_df = add_car_age(train_df)
val_df = add_car_age(val_df)
test_df = add_car_age(test_df)

print("=" * 70)
print(f"6.1 CAR AGE  (AGE_REFERENCE_YEAR = {AGE_REFERENCE_YEAR})")
print("=" * 70)
print("car_age distribution (Train):")
print(train_df["car_age"].describe())
n_negative_age = (train_df["car_age"] < 0).sum()
print(f"\ncar_age < 0 rows in Train: {n_negative_age} (expected 0 -- {AGE_REFERENCE_YEAR} "
      f"is the max year present in the cleaned dataset, so no row can be 'newer' than that)")
print()

# =====================================================================
# 6.2 Mileage Per Year
# =====================================================================
# car_age == 0 فعليًا بيحصل هنا (عربيات موديل 2020 -- آخر سنة في الداتا)،
# مش حالة نظرية بس. القسمة على صفر رياضيًا غير معرّفة -- مش معناها "صفر
# استخدام سنوي" (عربية جديدة بـ500 ميل مثلًا مش معناها معدل استخدامها صفر،
# معناها إننا مينفعش نحسب "في السنة" لعربية عمرها أقل من سنة). بنحطها NaN
# صراحة بدل ما نختراع رقم (0 أو غيره) -- نفس مبدأ التعامل مع engineSize في
# Part 3: نسيب القيمة غير المعروفة NaN بدل اختراعها.
def add_mileage_per_year(df):
    df = df.copy()
    df["mileage_per_year"] = np.where(
        df["car_age"] > 0,
        df["mileage"] / df["car_age"],
        np.nan,
    )
    return df


train_df = add_mileage_per_year(train_df)
val_df = add_mileage_per_year(val_df)
test_df = add_mileage_per_year(test_df)

n_zero_age_train = (train_df["car_age"] == 0).sum()
n_mpy_nan_train = train_df["mileage_per_year"].isna().sum()
print("=" * 70)
print("6.2 MILEAGE PER YEAR")
print("=" * 70)
print(f"car_age == 0 rows in Train: {n_zero_age_train} -> mileage_per_year set to NaN for "
      f"these (division by zero is undefined, NOT 'zero usage')")
print(f"mileage_per_year NaN count in Train: {n_mpy_nan_train} "
      f"(should equal the car_age==0 count above)")
print("\nmileage_per_year distribution (Train, NaN excluded):")
print(train_df["mileage_per_year"].describe())
print()

# ---- investigate the extremes: engineered features can surface outliers
# that the original Part 3 checks never looked for (Part 3 only flagged
# mileage > 200,000 -- never a LOW mileage relative to a car's age).
# Printing only -- no rows are removed here, same investigate-first
# principle as Part 3.
mpy_valid = train_df.dropna(subset=["mileage_per_year"])
print("Lowest 5 mileage_per_year rows in Train (possible data-entry oddities -- "
      "e.g. near-zero total mileage on an old car):")
print(mpy_valid.nsmallest(5, "mileage_per_year")[
    ["Make", "model", "year", "car_age", "mileage", "mileage_per_year", "price"]
].to_string(index=False))

print("\nHighest 5 mileage_per_year rows in Train (possible fleet/rental-style heavy "
      "early use, or a data-entry oddity):")
print(mpy_valid.nlargest(5, "mileage_per_year")[
    ["Make", "model", "year", "car_age", "mileage", "mileage_per_year", "price"]
].to_string(index=False))
print()

# =====================================================================
# 6.3 Premium Brand Flag  (Your Own Feature)
# =====================================================================
# ليه الـFeature ده: الـEDA في Part 4 (قسم 4.5) أظهر فجوة سعرية واضحة بين
# 3 مجموعات -- مرسيدس/أودي/BMW فوق الباقي بوضوح (متوسط > £22,500)، فولكس
# فاجن في النص (~£16,800)، والباقي أقل (~£10,000-12,700). عمود "Make" الخام
# عنده 9 قيم منفصلة (one-hot)، فالموديل بيحتاج يتعلم الترتيب ده من الصفر
# لكل براند لوحده. الـFeature ده بيدّي الموديل "تلميح" مباشر ومختصر عن
# المكانة السوقية العامة، وده نمط موثّق فعليًا في مشاريع تسعير سيارات
# مستعملة تانية (premium-brand flag جنب car age وmileage-per-year).
#
# مهم جدًا (تجنب Data Leakage من الأساس، مش بعد التأكد منه): التصنيف هنا
# مبني على معرفة خارجية عن وضع البراندات دي في سوق السيارات عمومًا (مرسيدس/
# أودي/BMW علامات ألمانية premium معروفة عالميًا، بغض النظر عن أي داتا) --
# مش محسوب من متوسط/median السعر في الـTrain set نفسه. ده الفرق الجوهري
# بين الـFeature ده والاقتراح البديل ("High Mileage Indicator" على median
# الـmileage_per_year): مفيش أي "fit" على الداتا هنا خالص، فمفيش خطر
# leakage من الأساس، والتصنيف ثابت ومش هيتغير لو اتغير حجم الـTrain set.
PREMIUM_MAKES = {"Mercedes", "Audi", "BMW"}


def add_premium_brand_flag(df):
    df = df.copy()
    df["is_premium_brand"] = df["Make"].isin(PREMIUM_MAKES).astype(int)
    return df


train_df = add_premium_brand_flag(train_df)
val_df = add_premium_brand_flag(val_df)
test_df = add_premium_brand_flag(test_df)

print("=" * 70)
print("6.3 PREMIUM BRAND FLAG")
print("=" * 70)
print(f"Premium makes (external market-positioning knowledge, NOT price-derived): "
      f"{sorted(PREMIUM_MAKES)}")
print("\nTrain -- rows per is_premium_brand:")
print(train_df["is_premium_brand"].value_counts())
print("\nTrain -- mean price by is_premium_brand (sanity check only -- NOT used to "
      "build the feature, the feature was defined before looking at this):")
print(train_df.groupby("is_premium_brand")["price"].mean().round(0))
print()

# =====================================================================
# 6.4 Save
# =====================================================================
train_df.to_csv("train_fe.csv", index=False)
val_df.to_csv("val_fe.csv", index=False)
test_df.to_csv("test_fe.csv", index=False)

print("=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)
print("New columns added to Train, Validation AND Test identically: "
      "car_age, mileage_per_year, is_premium_brand")
print(f"AGE_REFERENCE_YEAR used for car_age: {AGE_REFERENCE_YEAR}")
print(f"Premium makes used for is_premium_brand: {sorted(PREMIUM_MAKES)}")
print("All three features are pure per-row transforms or fixed external mappings -- "
      "none of them is fit on the Train set (no learned mean/median/threshold), so "
      "there is no data-leakage risk and they were computed independently and "
      "identically on Train, Validation and Test.")
print()
print("Saved: train_fe.csv, val_fe.csv, test_fe.csv")
