# =====================================================================
# PART 3: Data Cleaning
# =====================================================================
# مبدأ العمل في كل قسم: Detect -> Investigate -> Decide -> Document.
# القرارات هنا مبنية على النتائج اللي شفناها في الجزء التاني (الأعداد
# والـ breakdowns بالـ fuelType). مفيش أي حذف/تعديل من غير سبب موثّق
# جنب العملية نفسها في cleaning_log.

import numpy as np

combined_raw_backup = combined_cars.copy()   # نسخة من قبل أي تنظيف، للمقارنة لاحقًا
cleaning_log = []

rows_before = len(combined_cars)
print(f"Rows before cleaning: {rows_before}")

# تحقق فعلي، مش مجرد افتراض، إن الداتا داخلة نضيفة من Part 2
missing_before = combined_cars.isnull().sum().sum()
assert missing_before == 0, (
    f"Expected 0 missing values entering Part 3 (per Part 2), found {missing_before}. "
    f"Re-check Part 2 before continuing."
)
print(f"Missing values before cleaning: {missing_before} (verified above, not assumed)")

# =====================================================================
# 1) Duplicate Rows
# =====================================================================
# مفيش عمود ID فريد للإعلان/العربية، فمينفعش نتأكد 100% إن الصفوف دي
# نفس الإعلان اتكرر أو عربيتين مختلفتين صدفة بنفس الـ spec بالظبط.
# القرار (اجتهاد، مش حقيقة مؤكدة): نحذف الـ exact duplicates، لأنه حتى
# لو كانوا عربيات حقيقية مختلفة، الصف المكرر بيدّي نفس الـ combination
# وزن إضافي في تدريب موديل التسعير من غير معلومة جديدة فعلية.

dup_sample = combined_cars[combined_cars.duplicated(keep=False)].sort_values(
    list(combined_cars.columns)
)
print(f"\nExact duplicate rows: {combined_cars.duplicated().sum()}")
print("Sample:")
print(dup_sample.head(10))

before = len(combined_cars)
combined_cars = combined_cars.drop_duplicates().reset_index(drop=True)
dup_removed = before - len(combined_cars)
cleaning_log.append(
    f"Removed {dup_removed} exact duplicate rows. These were already present inside the "
    f"individual per-manufacturer files before combining -- the merge itself did not introduce "
    f"or multiply any duplicates (sum of per-file duplicate counts from Part 1 == this number). "
    f"No unique listing ID exists to prove these are the same ad scraped twice vs. two different "
    f"cars with identical specs -- this is a modeling judgment call, not a certainty, made to "
    f"avoid giving repeated feature combinations extra weight in a price model."
)
print(f"-> {len(combined_cars)} rows remain")

# =====================================================================
# 2) Invalid Years
# =====================================================================
# REFERENCE_YEAR بدل datetime.now().year (نفس المبدأ المستخدم في Part 2) --
# ثابت واحد معرّف في أول خلية بالنوتبوك، عشان النتيجة تفضل reproducible.
current_year = REFERENCE_YEAR

# ---- future years: treated as invalid, removed ----
invalid_year = combined_cars[combined_cars["year"] > current_year]
print(f"\n[Invalid year] rows with year > {current_year}: {len(invalid_year)}")
print(invalid_year)

before = len(combined_cars)
combined_cars = combined_cars[combined_cars["year"] <= current_year].reset_index(drop=True)
year_removed = before - len(combined_cars)
cleaning_log.append(
    f"Removed {year_removed} row(s) with year > {current_year} (Ford Fiesta, year=2060 -- clearly "
    f"invalid). NOT guessing a 'corrected' year, since there's no evidence for what the true "
    f"value should have been -- that would be fabricating data, not cleaning it."
)
print(f"-> {len(combined_cars)} rows remain")

# ---- very old years: بعد المراجعة، الصفوف دي سنتها مستحيلة تاريخيًا -> بتتحذف ----
# الصفين اللي year=1970 هما Mercedes M Class و Vauxhall Zafira. الموديلين دول نزلوا السوق
# في 1997 و1999، فالسنة المسجّلة مستحيلة (معلومة من تاريخ المنتج، مش مستنتجة من الداتا
# لوحدها). مفيش وسيلة موثوقة نعرف بيها السنة الصح، فالحذف أفضل من تخمين سنة بديلة.
# الـ assert تحت بيربط قرار الحذف بالموديلين اللي اتراجعوا بس: لو ظهر صف تاني
# بسنة قديمة (مثلًا عربية كلاسيك حقيقية) الكود هيقف عشان يتراجع بإيدك الأول.
LOW_YEAR_THRESHOLD = 1980
CONFIRM_REMOVE_VERY_OLD_YEARS = True

very_old = combined_cars[combined_cars["year"] < LOW_YEAR_THRESHOLD]
print(f"\n[Very old year] year < {LOW_YEAR_THRESHOLD}: {len(very_old)} rows:")
print(very_old.to_string())

very_old_removed = 0
if len(very_old) == 0:
    cleaning_log.append(f"Checked for year < {LOW_YEAR_THRESHOLD}: none found -- no action needed.")
elif CONFIRM_REMOVE_VERY_OLD_YEARS:
    unexpected_models = set(very_old["model"].str.strip()) - {"M Class", "Zafira"}
    assert not unexpected_models, (
        f"year < {LOW_YEAR_THRESHOLD} rows include models that were NOT part of the reviewed decision: "
        f"{unexpected_models}. Review them manually before removing."
    )
    removed_desc = "; ".join(
        f"{r.Make} {r.model.strip()} (year={r.year})" for r in very_old.itertuples()
    )
    before = len(combined_cars)
    combined_cars = combined_cars[combined_cars["year"] >= LOW_YEAR_THRESHOLD].reset_index(drop=True)
    very_old_removed = before - len(combined_cars)
    cleaning_log.append(
        f"Removed {very_old_removed} row(s) with year < {LOW_YEAR_THRESHOLD}: {removed_desc}. The recorded "
        f"model year predates the market introduction of the corresponding model lines (Mercedes M Class: "
        f"1997; Vauxhall Zafira: 1999 -- product-history knowledge, not derivable from this dataset alone), "
        f"so the year cannot be genuine. NOT guessing a 'corrected' year -- the true value is unknowable "
        f"from the data, and inventing one would be fabricating data, not cleaning it."
    )
    print(f"-> {len(combined_cars)} rows remain")
else:
    cleaning_log.append(
        f"Flagged {len(very_old)} row(s) with year < {LOW_YEAR_THRESHOLD} but did NOT remove them "
        f"(CONFIRM_REMOVE_VERY_OLD_YEARS = False). Left in place pending manual review."
    )
    print(f"-> {len(combined_cars)} rows remain (unchanged)")

# =====================================================================
# 3) Suspicious Engine Sizes (engineSize <= 0)
# =====================================================================
# لسيارات Petrol/Diesel/Hybrid: محرك احتراق بسعة 0 لتر مستحيل فيزيائيًا
# بحكم التعريف -- دي حقيقة ميكانيكية، مش تأويل. لـ Other فالتصنيف أقل
# يقينًا (فئة غامضة)، وبنعامله بنفس الطريقة مؤقتًا لحد ما نراجعه أكتر.
# ملحوظة: أي تفاصيل محددة عن صفوف بعينها لسه محتاجة تتراجع من بياناتك
# أنت مباشرة قبل ما تتاخد كحقيقة مؤكدة.

zero_engine = combined_cars[combined_cars["engineSize"] <= 0]
print(f"\n[engineSize <= 0] {len(zero_engine)} rows -- breakdown:")
print(zero_engine.groupby("fuelType").size())

is_zero_engine = combined_cars["engineSize"] <= 0
is_electric = combined_cars["fuelType"] == "Electric"
non_electric_zero = is_zero_engine & ~is_electric

engine_converted = int(non_electric_zero.sum())
combined_cars.loc[non_electric_zero, "engineSize"] = np.nan
cleaning_log.append(
    f"Converted engineSize=0 -> NaN for {engine_converted} non-Electric rows -- a combustion "
    f"engine cannot have 0L displacement by definition (Petrol/Diesel/Hybrid); 'Other' fuelType "
    f"is less certain and was treated the same way pending further review. Kept engineSize=0 for "
    f"Electric rows as-is (a standard convention for 'no combustion engine' in this kind of "
    f"dataset -- see the fuelType-vs-engineSize breakdown in section 9 for supporting context, "
    f"not asserted as an absolute claim about every individual row). NOT imputing a replacement "
    f"value here; that's a separate decision for the feature-engineering stage."
)
print(f"Converted {engine_converted} values to NaN. Remaining engineSize=0 (should be Electric only): "
      f"{(combined_cars['engineSize'] == 0).sum()}")

# =====================================================================
# 4) Unrealistic Prices
# =====================================================================
high_price = combined_cars[combined_cars["price"] > 100000].sort_values("price", ascending=False)
print(f"\n[price > 100000] {len(high_price)} rows -- full list "
      f"(this is an INVESTIGATION threshold, not a claim that every row above it is invalid):")
print(high_price)

# "A Class" + محرك 4.0L: ده مش مجرد سعر عالي -- ده تعارض في الـ spec نفسه.
# الدليل (من الداتا + مصدر خارجي، مش من السعر):
#   * من الداتا: كل A Class التانية في الداتاست engineSize بتاعها <= 2.1 -- الـ 4.0 كتلة
#     منفصلة تمامًا عن باقي التوزيع (شوف الـ groupby تحت).
#   * خارجيًا: Mercedes-AMG بتوثّق إن A 45 / A 45 S (أقوى A Class) محركها 2.0L رباعي
#     السلندر (1,991 cc) -- مفيش A Class متوثقة بمحرك 4.0L.
# المعيار هنا هو الـ engineSize مش السعر: الـ 17 اللي فوق £100k والـ 15 اللي تحت £100k
# نفس المشكلة بالظبط، فحذف الـ 17 بس كان هيسيب 15 صف فيهم نفس التناقض.
# مش بنغيّر engineSize لـ 2.0 ومش بنغيّر model -- مفيش وسيلة نعرف بيها أنهي عمود هو الغلط
# (ممكن listing لموديل AMG تاني اتصنّف غلط كـ A Class)، والتصحيح بالتخمين اختراع بيانات.

A_CLASS_MAX_PLAUSIBLE_ENGINE = 2.1

a_class_all = combined_cars[combined_cars["model"].str.strip() == "A Class"]
a_class_price_stats = a_class_all["price"].describe()
print(f"\nContext -- price distribution for ALL {len(a_class_all)} 'A Class' listings in the dataset:")
print(a_class_price_stats)

print("\n'A Class': price by engineSize (is 4.0 a separate cluster from every other engine size?):")
print(a_class_all.groupby("engineSize")["price"].agg(["count", "min", "median", "max"]))

a_class_bad_engine = a_class_all[a_class_all["engineSize"] > A_CLASS_MAX_PLAUSIBLE_ENGINE]
n_bad_expensive = int((a_class_bad_engine["price"] > 100000).sum())
n_bad_other = len(a_class_bad_engine) - n_bad_expensive
print(f"\n[A Class + engineSize > {A_CLASS_MAX_PLAUSIBLE_ENGINE}] {len(a_class_bad_engine)} rows out of "
      f"{len(a_class_all)} 'A Class' listings ({n_bad_expensive} priced > £100,000, "
      f"{n_bad_other} priced <= £100,000) -- full list:")
print(a_class_bad_engine.sort_values("price", ascending=False).to_string(index=False))

# لو راجعت الصفوف وقررت تبقيها، غيّر القيمة تحت لـ False وشغّل الكود تاني.
CONFIRM_REMOVE_A_CLASS_ENGINE_MISMATCH = True

a_class_flagged = len(a_class_bad_engine)
a_class_removed = 0   # بيتحدّث تحت لو الحذف اتنفّذ فعليًا

if CONFIRM_REMOVE_A_CLASS_ENGINE_MISMATCH and a_class_flagged > 0:
    before = len(combined_cars)
    bad_engine_mask = (
        (combined_cars["model"].str.strip() == "A Class")
        & (combined_cars["engineSize"] > A_CLASS_MAX_PLAUSIBLE_ENGINE)
    )
    combined_cars = combined_cars[~bad_engine_mask].reset_index(drop=True)
    a_class_removed = before - len(combined_cars)
    cleaning_log.append(
        f"Removed {a_class_removed} 'A Class' row(s) with engineSize > {A_CLASS_MAX_PLAUSIBLE_ENGINE}L "
        f"({n_bad_expensive} priced above £100,000 and {n_bad_other} at or below it). The criterion is "
        f"the model/engine combination, NOT price: every other 'A Class' listing in the dataset is <= "
        f"{A_CLASS_MAX_PLAUSIBLE_ENGINE}L, and Mercedes-AMG documents the A 45 / A 45 S (the most "
        f"powerful A-Class) as a 2.0L four-cylinder (1,991 cc), so a 4.0L A-Class contradicts the "
        f"documented specifications. NOT 'fixed' (e.g. 4.0 -> 2.0, or relabelling the model): the "
        f"dataset cannot show whether 'model' or 'engineSize' is the wrong field (possibly a different "
        f"AMG model listed as 'A Class'), and a guessed correction would be fabricated data. Removed "
        f"rather than kept because the row cannot be trusted as a (model, spec, price) example for a "
        f"price model. High prices in other models (G Class, R8, SL Class, ...) are NOT treated as errors."
    )
    print(f"-> {len(combined_cars)} rows remain")
else:
    cleaning_log.append(
        f"'A Class' rows with engineSize > {A_CLASS_MAX_PLAUSIBLE_ENGINE}L ({a_class_flagged} found) "
        f"contradict documented A-Class specifications but were NOT removed "
        f"(CONFIRM_REMOVE_A_CLASS_ENGINE_MISMATCH = False). Left in place pending manual review."
    )
    print("Not removed (CONFIRM_REMOVE_A_CLASS_ENGINE_MISMATCH = False).")

# باقي الأسعار العالية (مش A Class) -- بتتراجع وتتسجل بنفس المبدأ، مش
# مجرد طباعة من غير قرار موثّق.
other_high_price = combined_cars[
    (combined_cars["price"] > 100000) & (combined_cars["model"].str.strip() != "A Class")
]
print(f"\nOther models priced > £100,000 (excluding A Class): {len(other_high_price)} rows:")
print(other_high_price[["Make", "model", "year", "price", "mileage"]].sort_values(
    "price", ascending=False
).to_string(index=False))
cleaning_log.append(
    f"Kept {len(other_high_price)} non-'A Class' row(s) priced above £100,000 (e.g. G Class, R8, "
    f"SL Class) as-is -- plausible for those specific luxury/performance models, no pattern "
    f"suggesting a data-entry error the way the 'A Class' + 4.0L cluster does."
)

low_price = combined_cars[combined_cars["price"] < 500]
print(f"\n[price < 500] {len(low_price)} rows (kept -- old, high-mileage cars, plausible):")
print(low_price)
cleaning_log.append(
    f"Kept {len(low_price)} price < 500 row(s) as-is -- old cars with high mileage, consistent with "
    f"genuinely low resale value. No evidence of error."
)

# =====================================================================
# 5) Unrealistic Mileage
# =====================================================================
high_mileage = combined_cars[combined_cars["mileage"] > 200000]
print(f"\n[mileage > 200000] {len(high_mileage)} rows (kept -- plausible for old, high-use cars):")
print(high_mileage)
cleaning_log.append(
    f"Kept {len(high_mileage)} mileage > 200,000 row(s) as-is -- high mileage is possible for "
    f"older used vehicles, and no direct evidence of a data-entry error was found (prices for "
    f"these rows are consistently low, which is consistent with genuine high-mileage cars rather "
    f"than suspicious)."
)

# =====================================================================
# 6) Inconsistent Categories
# =====================================================================
# بندور بس على تعارض حقيقي: نفس القيمة بعد التطبيع (strip + lower) بتترجم
# لأكتر من صيغة مختلفة في العمود الأصلي (زي "Automatic " و"Automatic").
# ده أدق من مجرد النظر لقائمة unique() بالعين، وأأمن من .str.title() على
# الكل (ده كان هيكسر أسماء زي "i3" أو "GT86").

def find_inconsistent_labels(df, col):
    normalized = df[col].astype(str).str.strip().str.lower()
    groups = df.groupby(normalized)[col].unique()
    return {k: v for k, v in groups.items() if len(v) > 1}

for col in ["model", "transmission", "fuelType", "Make"]:
    issues = find_inconsistent_labels(combined_cars, col)
    if issues:
        print(f"\nPossible inconsistencies in '{col}':")
        for variants in issues.values():
            print(f"  {list(variants)}")
    else:
        print(f"\nNo colliding spelling/spacing/case variants found in '{col}'.")
# ملحوظة: "SL CLASS" (كابيتال كامل) مقابل "A Class"/"G Class" (Title Case)
# ده اختلاف أسلوب كتابة، مش بالضرورة "تعارض" بالمعنى اللي الدالة دي بتدور
# عليه -- إلا لو فيه صيغة تانية زي "SL Class" موجودة في مكان تاني وبتتعارض معاها.

# =====================================================================
# 7) MPG (مش مطلوبة صراحة في التعليمات، لكن اتكشفت في الجزء التاني)
# =====================================================================
high_mpg = combined_cars[combined_cars["mpg"] > 200]
print(f"\n[mpg > 200] {len(high_mpg)} rows -- breakdown:")
print(high_mpg.groupby("fuelType").size())

# إضافة: هل أعلى قيمة mpg في الداتاست كلها بتتكرر بنفس الرقم بالظبط عبر أكتر
# من fuelType؟ (الفحص ده هو اللي كشف إن الرقم ده مش عيب بيانات -- هو الرقم
# الرسمي NEDC Combined بتاع BMW i3 Range Extender، اتأكد بمصدر خارجي).
max_mpg = combined_cars["mpg"].max()
at_max_mpg = combined_cars[combined_cars["mpg"] == max_mpg]
groups_at_max = sorted(at_max_mpg["fuelType"].unique())
print(f"\n[mpg == dataset max ({max_mpg})] {len(at_max_mpg)} rows -- breakdown by fuelType:")
print(at_max_mpg["fuelType"].value_counts())

# الفيصل بين احتمالين: (أ) القيمة دي rated spec رسمية مشتركة لموديل/فئة PHEV
# معينة (طبيعي -- mpg هنا مش قياس مستقل لكل عربية)، أو (ب) sentinel/capped
# value غريبة. النتيجة (شوف الـcleaning_log تحت): احتمال (أ) هو الصح، 43/43
# صف كلهم موديل "i3" -- مؤكد خارجيًا إنه رقم BMW الرسمي، مش خطأ بيانات.
print(f"\n[mpg == dataset max ({max_mpg})] breakdown by model (checks whether this is a "
      f"shared official spec value for one/few models, vs. scattered across many):")
print(at_max_mpg["model"].value_counts())

n_models_at_max = at_max_mpg["model"].nunique()
top_model_share = at_max_mpg["model"].value_counts().iloc[0] / len(at_max_mpg)

cleaning_log.append(
    f"Kept {len(high_mpg)} mpg > 200 row(s) as-is -- concentrated in Hybrid/Electric/Other, matching "
    f"the official UK NEDC 'Combined' mpg methodology for plug-in/range-extender hybrids (the test "
    f"cycle assumes a mostly-electric run on a full battery, which inflates the petrol-equivalent "
    f"combined mpg figure far beyond real-world use). RESOLVED (not just flagged): the single highest "
    f"mpg value in the dataset ({max_mpg}) recurs in {len(at_max_mpg)} row(s) across "
    f"{len(groups_at_max)} fuelType label(s) ({', '.join(groups_at_max)}), but ALL {len(at_max_mpg)} "
    f"of them are model '{at_max_mpg['model'].value_counts().index[0]}' ({top_model_share:.0%} of the "
    f"subset, {n_models_at_max} distinct model overall). Cross-checked against independent external "
    f"sources (Autocar, Honest John/RealMPG, CAR Magazine): {max_mpg} mpg is BMW's own official "
    f"published NEDC Combined rating for the BMW i3 Range Extender -- confirmed to be a genuine "
    f"manufacturer-published spec value, NOT a data-entry error or a capped/sentinel artifact. "
    f"mpg/tax/engineSize in this dataset are per-model/trim spec values rather than independently "
    f"measured per vehicle, which is exactly why all i3 REx listings share it identically. Kept as-is "
    f"with confidence; not removed."
)

# =====================================================================
# 8) Logical Consistency / Integrity Checks (sanity net)
# =====================================================================
integrity_checks = {
    "price <= 0":     (combined_cars["price"] <= 0).sum(),
    "mileage < 0":    (combined_cars["mileage"] < 0).sum(),
    "tax < 0":        (combined_cars["tax"] < 0).sum(),
    "mpg <= 0":       (combined_cars["mpg"] <= 0).sum(),
    "engineSize < 0": (combined_cars["engineSize"] < 0).sum(),
}
print("\nIntegrity checks (expected all 0):")
for label, count in integrity_checks.items():
    print(f"  {label}: {count}")

# =====================================================================
# 9) Cross-Feature Consistency Checks (investigation only --
#    no rows are removed here; findings may justify revisiting
#    decisions made above before moving on to EDA)
# =====================================================================

# year vs mileage -- a near-new car with implausibly high mileage
new_but_high_mileage = combined_cars[
    (combined_cars["year"] >= current_year - 2) & (combined_cars["mileage"] > 100000)
]
print(f"\n[year >= {current_year-2} AND mileage > 100,000] {len(new_but_high_mileage)} rows:")
print(new_but_high_mileage)

# year vs price -- a very old car at a very high price
old_but_expensive = combined_cars[
    (combined_cars["year"] <= 2005) & (combined_cars["price"] > 30000)
]
print(f"\n[year <= 2005 AND price > £30,000] {len(old_but_expensive)} rows:")
print(old_but_expensive)

# fuelType vs engineSize -- is Electric=0 really a clean convention here,
# or does engineSize vary for EVs too (relevant to the decision in step 3)?
print("\nengineSize distribution BY fuelType:")
print(combined_cars.groupby("fuelType")["engineSize"].describe())

# fuelType vs mpg -- context for the mpg > 200 decision in step 7
print("\nmpg distribution BY fuelType:")
print(combined_cars.groupby("fuelType")["mpg"].describe())

# transmission vs model -- is 'Other' transmission concentrated in specific models?
print("\ntransmission value counts:")
print(combined_cars["transmission"].value_counts())
print("\nModels most associated with transmission == 'Other':")
print(combined_cars[combined_cars["transmission"] == "Other"]["model"].value_counts().head(10))

cleaning_log.append(
    "Ran cross-feature consistency checks (year-vs-mileage, year-vs-price, fuelType-vs-engineSize, "
    "fuelType-vs-mpg, transmission-vs-model) as an investigation pass -- no rows were removed based "
    "on these alone. Review the printed breakdowns above; findings may justify revisiting the "
    "Electric-engineSize, MPG>200, or A-Class decisions above before moving on to EDA."
)

# =====================================================================
# Final Verification + Cleaning Log
# =====================================================================
print(f"\nCurrent year used for validation: {current_year}")
print(f"\nFinal shape: {combined_cars.shape} "
      f"(removed {len(combined_raw_backup) - len(combined_cars)} rows total from the raw combine)")

remaining_missing = combined_cars.isnull().sum()
remaining_missing_cols = set(remaining_missing[remaining_missing > 0].index)
print("\nMissing values after cleaning:")
print(remaining_missing[remaining_missing > 0] if remaining_missing_cols else "None found.")

# تحقق فعلي إن مفيش missing values ظهرت في عمود غير المتوقع (engineSize)
assert remaining_missing_cols <= {"engineSize"}, (
    f"Unexpected missing values outside engineSize: {remaining_missing_cols - {'engineSize'}}. "
    f"Investigate before continuing to Part 4."
)
print("(Confirmed: the only remaining missing values are the deliberate engineSize NaNs from "
      "step 3 -- left unimputed on purpose.)")

# فحوصات رجوعية (regression checks) -- تأكيد رقمي إن كل خطوة اتنفذت صح،
# مش مجرد افتراض إن الكود اشتغل.
print(f"\nDuplicates remaining: {combined_cars.duplicated().sum()}")
print(f"Invalid future years remaining (year > {current_year}): "
      f"{(combined_cars['year'] > current_year).sum()}")
print(f"Non-Electric invalid engine sizes remaining: "
      f"{((combined_cars['engineSize'] <= 0) & (combined_cars['fuelType'] != 'Electric')).sum()}")
print(f"Very-old years remaining (year < {LOW_YEAR_THRESHOLD}): "
      f"{(combined_cars['year'] < LOW_YEAR_THRESHOLD).sum()}")
_is_a_class = combined_cars["model"].str.strip() == "A Class"
print(f"'A Class' rows with engineSize > {A_CLASS_MAX_PLAUSIBLE_ENGINE} remaining: "
      f"{(_is_a_class & (combined_cars['engineSize'] > A_CLASS_MAX_PLAUSIBLE_ENGINE)).sum()}")
print(f"'A Class' rows priced > £100,000 remaining: "
      f"{(_is_a_class & (combined_cars['price'] > 100000)).sum()}")

print("\n" + "=" * 70)
print("CLEANING LOG (narrative -- why each decision was made)")
print("=" * 70)
for i, entry in enumerate(cleaning_log, 1):
    print(f"{i}. {entry}")

# ---- ملخص رقمي مختصر (ما حدث فعليًا، مش المخطط) ----
final_rows = len(combined_cars)
final_missing_engine = int(combined_cars["engineSize"].isnull().sum())

print("\n" + "=" * 70)
print("CLEANING SUMMARY (counts -- reflects what actually happened)")
print("=" * 70)
summary_rows = [
    ("Rows before cleaning",                                      rows_before),
    ("Exact duplicate rows removed",                               dup_removed),
    ("Invalid future-year rows removed",                           year_removed),
    (f"Impossible very-old-year rows removed (year < {LOW_YEAR_THRESHOLD})", very_old_removed),
    ("Non-Electric engineSize values converted to NaN",            engine_converted),
    (f"'A Class' impossible-engine (> {A_CLASS_MAX_PLAUSIBLE_ENGINE}L) rows removed", a_class_removed),
    ("'A Class' impossible-engine rows flagged (not removed)",     a_class_flagged - a_class_removed),
    ("Other high-price rows kept (reviewed, plausible)",           len(other_high_price)),
    ("Low-price rows kept (reviewed, plausible)",                  len(low_price)),
    ("High-mileage rows kept (reviewed, plausible)",               len(high_mileage)),
    ("High-mpg rows kept (resolved: official NEDC spec, not an error)", len(high_mpg)),
    ("Rows after cleaning",                                        final_rows),
    ("Missing values remaining (engineSize only)",                 final_missing_engine),
]
label_width = max(len(label) for label, _ in summary_rows) + 2
for label, value in summary_rows:
    print(f"{label:<{label_width}} {value}")
print("=" * 70)

combined_cars.to_csv("combined_cars_clean.csv", index=False)
print("\nSaved cleaned dataset to combined_cars_clean.csv")
