# =====================================================================
# PART 14: Build the Smart Deal Advisor
# =====================================================================
# Input : deal_advisor.py (the advisor logic, same folder), part12_final_model.json, models/*.joblib,
#         preprocessor.joblib, preprocessed_data.joblib, train_fe.csv, val_fe.csv (Part 6 / 7 / 12),
#         test_predictions.csv (Part 12, used only for the descriptive section 14.7)
# Output: part14_examples.csv, part14_test_rating_distribution.csv
#
# Rules:
#   1. No training and no tuning. The advisor is a decision layer on top of the model chosen in Part 12.
#   2. The rating rules are fixed in deal_advisor.py (from the assignment) before any Test number is shown.
#   3. The Parity test (14.4) uses Validation rows only. If it fails, the script stops: the advisor would be
#      building features differently from Part 6 / 7 and every later number would be wrong.
#   4. The Test set is used ONLY in 14.7, to describe how often the rules fire. It is not used to change
#      thresholds, features, preprocessing or the model.

import json

import joblib
import numpy as np
import pandas as pd

import deal_advisor as da
from deal_advisor import AdvisorInputError, DealAdvisor, format_report, get_deal_rating

PARITY_ROWS = 20
RANDOM_STATE = 42
EXAMPLES_FILE = "part14_examples.csv"
TEST_DIST_FILE = "part14_test_rating_distribution.csv"


def banner(title):
    print("=" * 90)
    print(title)
    print("=" * 90)


# ----------------------------------------------------------------------
# 14.1 Configuration + load the advisor
# ----------------------------------------------------------------------
banner("14.1 CONFIGURATION")
adv = DealAdvisor()
print(f"Final model (read from {da.DECISION_FILE}): {adv.model_name}   |   preprocessor: {da.PREPROCESSOR_FILE}")
print(f"Reference year for car_age: {da.AGE_REFERENCE_YEAR} | premium makes: {sorted(da.PREMIUM_MAKES)}  "
      "(both verified against train_fe.csv)")
print("Deal rating, below_market_pct = (market - seller) / market x 100, rounded to 1 decimal first:")
print(f"   > {da.GREAT_ABOVE:g}%  Great Deal | {da.GOOD_FROM:g} to {da.GREAT_ABOVE:g}%  Good Deal | "
      f"between {da.FAIR_ABOVE:g}% and {da.GOOD_FROM:g}%  Fair Price | "
      f"{da.SLIGHTLY_FROM:g} to {da.FAIR_ABOVE:g}%  Slightly Overpriced | < {da.SLIGHTLY_FROM:g}%  Overpriced")
print("   (the assignment does not say which side the exact boundaries belong to; this rule is symmetric)")
print(f"Reliability notes (operational, from Part 13): estimate < £{da.LOW_PRICE_WARN:,} or > £{da.HIGH_PRICE_WARN:,}, "
      f"fewer than {da.RARE_MODEL_N} Train cars of that Make+Model, MPG/Tax filled in, inputs outside Train range.")
print()

# ----------------------------------------------------------------------
# 14.2 Deal rating: unit tests (pure business rules, no model, no data)
# ----------------------------------------------------------------------
banner("14.2 DEAL RATING FUNCTION -- UNIT TESTS")
cases = [  # (market, seller, expected pct, expected rating)
    (20000, 16000, 20.0, "Great Deal"),
    (20000, 17900, 10.5, "Great Deal"),
    (20000, 18000, 10.0, "Good Deal"),            # exactly 10 is not "more than 10%"
    (20000, 19000, 5.0, "Good Deal"),
    (20000, 19100, 4.5, "Fair Price"),
    (20000, 20000, 0.0, "Fair Price"),
    (20000, 20900, -4.5, "Fair Price"),
    (20000, 21000, -5.0, "Slightly Overpriced"),
    (20000, 22000, -10.0, "Slightly Overpriced"),
    (20000, 22100, -10.5, "Overpriced"),
    (10000, 9504, 5.0, "Good Deal"),              # 4.96% is shown as 5.0% -> must be rated as 5.0%
    (18500, 17000, 8.1, "Good Deal"),             # the assignment's example
]
print(f"{'Market':>9s} {'Seller':>9s} {'below %':>8s}  Rating")
for market, seller, exp_pct, exp_rating in cases:
    r = get_deal_rating(market, seller)
    assert r["below_market_pct"] == exp_pct, (market, seller, r["below_market_pct"], exp_pct)
    assert r["deal_rating"] == exp_rating, (market, seller, r["deal_rating"], exp_rating)
    print(f"{'£' + format(market, ','):>9s} {'£' + format(seller, ','):>9s} {r['below_market_pct']:>+8.1f}  "
          f"{r['deal_icon']} {r['deal_rating']}   ({r['difference_text']})")
assert get_deal_rating(18500, 17000)["difference_text"] == "£1,500 cheaper"
assert get_deal_rating(20000, 20500)["difference_text"] == "£500 more expensive"
assert get_deal_rating(20000, 20000)["difference_text"] == "same as the estimated market price"
for bad_pred, bad_sell in [(20000, 0), (20000, -500), (20000, float("nan")), (20000, float("inf")),
                           (20000, None), (20000, "abc"), (20000, True), (0, 17000), (-1, 17000),
                           (float("nan"), 17000)]:
    try:
        get_deal_rating(bad_pred, bad_sell)
    except ValueError:
        continue
    raise AssertionError(f"invalid prices were accepted: {bad_pred!r}, {bad_sell!r}")
print("OK: all boundary cases and invalid-price cases behave as specified.")
print()

# ----------------------------------------------------------------------
# 14.3 Input validation (a valid base car is taken from the Train data)
# ----------------------------------------------------------------------
banner("14.3 INPUT VALIDATION")
pair = adv.pair_counts.idxmax()
src = adv.train[(adv.train["Make"] == pair[0]) & (adv.train["model"] == pair[1])
                & adv.train["engineSize"].notna()].iloc[0]
base = {"make": src["Make"], "model": src["model"], "year": int(src["year"]), "mileage": float(src["mileage"]),
        "transmission": src["transmission"], "fuel_type": src["fuelType"], "engine_size": float(src["engineSize"]),
        "mpg": float(src["mpg"]), "tax": float(src["tax"])}
print(f"Base car for the tests (a real Train row): {base}")
est_base = adv.estimate(base)
print(f"   -> estimate £{est_base['predicted_price']:,.0f}")

bad_inputs = {
    "unknown make": {**base, "make": "Tesla"},
    "model of another make": {**base, "make": "Ford" if base["make"] != "Ford" else "BMW"},
    "unknown transmission": {**base, "transmission": "CVT-X"},
    "unknown fuel type": {**base, "fuel_type": "Steam"},
    "year newer than the data": {**base, "year": da.AGE_REFERENCE_YEAR + 3},
    "fractional year": {**base, "year": 2015.5},
    "negative mileage": {**base, "mileage": -1},
    "engine size 0": {**base, "engine_size": 0},
    "mpg not a number": {**base, "mpg": "lots"},
    "missing year": {k: v for k, v in base.items() if k != "year"},
    "year is NaN": {**base, "year": float("nan")},
}
for label, bad in bad_inputs.items():
    try:
        adv.estimate(bad)
    except AdvisorInputError as e:
        print(f"   rejected [{label}]: {e.errors[0][:100]}")
    else:
        raise AssertionError(f"invalid input accepted: {label}")
for bad_seller in [0, -100, float("nan"), None]:
    try:
        adv.advise(base, bad_seller)
    except ValueError:
        continue
    raise AssertionError(f"seller price {bad_seller!r} was accepted")

messy = {**base, "make": f"  {base['make'].lower()} ", "model": f" {base['model'].upper()}  ",
         "transmission": base["transmission"].lower(), "fuel_type": base["fuel_type"].upper()}
assert adv.estimate(messy)["predicted_price"] == est_base["predicted_price"], "case/space handling changed the result"
print("OK: invalid cars/prices are rejected with a clear message; case and surrounding spaces are forgiven.")
print()

# ----------------------------------------------------------------------
# 14.4 PARITY TEST: raw Validation rows through the advisor == the Part 7 matrix + final model
# ----------------------------------------------------------------------
banner("14.4 PARITY TEST  (Validation rows only; stops the script if it fails)")
data = joblib.load("preprocessed_data.joblib")
X_val = data["X_val"]
val_raw = pd.read_csv("val_fe.csv")
assert len(val_raw) == len(X_val), "val_fe.csv and X_val have different row counts"

rng = np.random.default_rng(RANDOM_STATE)
special = list(val_raw.index[val_raw["car_age"] == 0][:3]) + list(val_raw.index[val_raw["engineSize"].isna()][:3])
others = [i for i in rng.permutation(len(val_raw)) if i not in special]
positions = sorted(set(special + others[:max(0, PARITY_ROWS - len(special))]))
n_age0 = int((val_raw.loc[positions, "car_age"] == 0).sum())
n_eng_nan = int(val_raw.loc[positions, "engineSize"].isna().sum())
print(f"{len(positions)} Validation rows ({n_age0} with car_age 0 -> mileage_per_year NaN, "
      f"{n_eng_nan} with no engine size).")

max_diff, n_full_path, n_full_rejected = 0.0, 0, 0
for i in positions:
    r = val_raw.iloc[i]
    clean = {c: (str(r[c]).strip() if c in da.CATEGORICAL else r[c])
             for c in ["Make", "model", "year", "mileage", "transmission", "fuelType", "tax", "mpg", "engineSize"]}
    feats = adv.assemble_features(clean)
    for c in ["car_age", "mileage_per_year", "is_premium_brand"]:       # engineered features == Part 6 columns
        assert np.isclose(float(feats.loc[0, c]), float(r[c]), equal_nan=True), f"row {i}: {c} differs from Part 6"
    processed = adv.preprocessor.transform(feats)                        # matrix == Part 7 matrix
    assert list(processed.columns) == list(X_val.columns), f"row {i}: columns differ from X_val"
    assert np.allclose(processed.to_numpy(float), X_val.iloc[[i]].to_numpy(float), equal_nan=True), \
        f"row {i}: preprocessed values differ from X_val"
    p_adv = adv.predict_price(feats)
    p_ref = float(adv.model.predict(X_val.iloc[[i]])[0])
    assert np.isclose(p_adv, p_ref, rtol=1e-9, atol=1e-6), f"row {i}: advisor {p_adv} != model {p_ref}"
    max_diff = max(max_diff, abs(p_adv - p_ref))

    # full public path (validation + imputation logic) for the rows it accepts
    raw_in = {"make": clean["Make"], "model": clean["model"], "year": clean["year"], "mileage": clean["mileage"],
              "transmission": clean["transmission"], "fuel_type": clean["fuelType"],
              "engine_size": None if pd.isna(clean["engineSize"]) else clean["engineSize"],
              "mpg": clean["mpg"], "tax": clean["tax"]}
    try:
        est = adv.estimate(raw_in)
    except AdvisorInputError:
        n_full_rejected += 1
        continue
    n_full_path += 1
    if not est["imputed"]:
        assert est["predicted_price"] == float(round(p_ref)), f"row {i}: estimate() differs from the model"
print(f"OK: engineered features, preprocessed matrix and predictions match Part 6 / 7 / 12 on all "
      f"{len(positions)} rows (largest price difference £{max_diff:.2e}).")
print(f"    Public path (validation + estimate): {n_full_path} rows accepted and identical, "
      f"{n_full_rejected} rejected by validation (e.g. a Make+Model unseen in Train, or an engine size that "
      "Part 3 left empty for a fuel type where most cars do have one).")
print()

# ----------------------------------------------------------------------
# 14.5 Examples
# ----------------------------------------------------------------------
banner("14.5 SMART DEAL EXAMPLES")
example_rows = []


def run_example(label, raw, seller_price):
    print(f"--- {label}")
    try:
        res = adv.advise(raw, seller_price)
    except AdvisorInputError as e:
        print("   Could not value this car:")
        for err in e.errors:
            print(f"   - {err[:200]}")
        print()
        return None
    print(format_report(res))
    print()
    example_rows.append({"example": label, **{k: v for k, v in res.items()
                                              if k not in ("car", "imputed", "warnings", "reliability_reasons")},
                         "reliability_reasons": " | ".join(res["reliability_reasons"])})
    return res


bmw = {"make": "BMW", "model": "3 Series", "year": 2019, "mileage": 35000, "transmission": "Automatic",
       "fuel_type": "Petrol", "engine_size": 2.0, "mpg": 45.0, "tax": 150}
try:
    market = adv.estimate(bmw)["predicted_price"]
except AdvisorInputError as e:
    print(f"The illustrative BMW is not valid for this dataset ({e.errors[0][:120]}); using the Train base car.")
    bmw, market = base, est_base["predicted_price"]
print(f"Same car, five asking prices around the estimate of £{market:,.0f} (illustrative inputs, not real listings):")
print()
for factor in [0.80, 0.93, 1.00, 1.07, 1.20]:
    run_example(f"asking price = {factor:.0%} of the estimate", bmw, round(market * factor / 50) * 50)

no_mpg_tax = {k: v for k, v in bmw.items() if k not in ("mpg", "tax")}
run_example("MPG and Tax left empty", no_mpg_tax, round(market * 0.93 / 50) * 50)
run_example("expensive, rare car",
            {"make": "Mercedes", "model": "G Class", "year": 2019, "mileage": 12000, "transmission": "Automatic",
             "fuel_type": "Petrol", "engine_size": 4.0, "mpg": 22.0, "tax": 580}, 120000)
run_example("invalid: year newer than the data", {**bmw, "year": 2024}, 20000)
if example_rows:
    pd.DataFrame(example_rows).to_csv(EXAMPLES_FILE, index=False)

# ----------------------------------------------------------------------
# 14.6 How the reliability notes are decided
# ----------------------------------------------------------------------
banner("14.6 RELIABILITY NOTES")
print("The deal rating and the reliability note are independent:")
print(" - the RATING only compares the asking price with the model's estimate;")
print(" - the RELIABILITY note says when that estimate deserves more care (see the thresholds in 14.1).")
print("The £5k / £40k limits come from the Part 13 error-by-band tables. They are practical warning levels, not")
print("statistical limits, and they are fixed in deal_advisor.py. No 'confidence %' is shown: R² is not a")
print("confidence level, and a Random Forest prediction has no probability attached to it.")
print()

# ----------------------------------------------------------------------
# 14.7 Test set, DESCRIPTIVE ONLY: how often would each rating appear if seller price == true price?
# ----------------------------------------------------------------------
banner("14.7 RATING DISTRIBUTION ON TEST  (descriptive only; nothing is changed because of it)")
pred_file = pd.read_csv("test_predictions.csv")
rows = []
for actual, predicted in zip(pred_file["y_true"], pred_file["y_pred"]):
    r = get_deal_rating(float(round(predicted)), float(actual))
    rows.append((r["deal_rating"], float(round(predicted))))
dist = pd.DataFrame(rows, columns=["rating", "predicted"])
order = [v[0] for v in da.RATING_LABELS.values()]
overall = dist["rating"].value_counts().reindex(order, fill_value=0)
bands = pd.cut(dist["predicted"], [0, da.LOW_PRICE_WARN, da.HIGH_PRICE_WARN, np.inf], right=False,
               labels=[f"< £{da.LOW_PRICE_WARN:,}", f"£{da.LOW_PRICE_WARN:,}-£{da.HIGH_PRICE_WARN:,}",
                       f"≥ £{da.HIGH_PRICE_WARN:,}"])
table = pd.crosstab(dist["rating"], bands).reindex(order, fill_value=0)
table["All Test"] = overall
pct_table = (table / table.sum() * 100).round(1)
print("Seller price := the car's TRUE price (a stand-in: Test has no real asking prices). A perfect model would")
print("rate every car 'Fair Price', so any other rating here is caused by model error alone.")
print()
print("Share of cars per rating (%), by predicted-price band:")
print(pct_table.to_string())
print()
print(f"N per band: {table.drop(index=[], errors='ignore').sum().astype(int).to_dict()}")
off = 100 - pct_table.loc["Fair Price", "All Test"]
print(f"=> Even with a perfectly honest price, {off:.1f}% of Test cars would be rated something other than "
      "'Fair Price' because of model error.")
print("   Compare with Part 12: about 47% of Test cars are within ±5% and 77% within ±10% of their true price.")
print("   Read a rating next to the reliability note, and treat 'Great Deal' / 'Overpriced' as 'worth checking'.")
table.to_csv(TEST_DIST_FILE)
print()

# ----------------------------------------------------------------------
# 14.8 Log
# ----------------------------------------------------------------------
banner("PART 14 LOG")
print(f"Advisor = raw input -> validation -> Part 6 features -> Part 7 preprocessor -> {adv.model_name} "
      "(from Part 12) -> rating.")
print("No model was trained or changed. The Parity test (Validation) proves the advisor rebuilds the Part 7")
print("matrix and reproduces the model's predictions exactly. Test was used only in 14.7, descriptively.")
print()
print("Limitations to write in your own words:")
print(" 1. The estimate is a model output from historical listings; condition, location, service history,")
print("    trim and optional extras are not in the data.")
print(f"2. The data ends at model year {da.AGE_REFERENCE_YEAR}: newer cars are rejected, not guessed.")
print(" 3. Errors are larger for cheap and for expensive/rare cars (Part 13); the rating thresholds (±5%, ±10%)")
print("    are of the same size as the typical model error (Part 12), so the rating is a screening signal.")
print(" 4. MPG / Tax left empty are replaced by Train medians of similar cars, and the report says so.")
print(" 5. The advisor can only value Make + Model combinations seen in the training data.")
print()
print(f"Saved: {EXAMPLES_FILE}, {TEST_DIST_FILE}. Module for Part 15: deal_advisor.py "
      "(from deal_advisor import DealAdvisor, format_report).")
