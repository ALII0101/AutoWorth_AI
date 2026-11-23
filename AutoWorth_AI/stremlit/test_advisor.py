# =====================================================================
# test_advisor.py -- checks that AutoWorth AI works correctly on YOUR machine
# =====================================================================
# Run from the project folder (same place as app.py):
#       .\.venv\Scripts\python.exe test_advisor.py
# It trains nothing and uses no Test data. Exit code 0 = everything passed.

import os
import sys
import json
import warnings

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ----------------------------------------------------------------------
print("\n== 1. Environment ==")
check("scikit-learn is 1.6.1 (the version the .joblib files were saved with)",
      sklearn.__version__ == "1.6.1", f"found {sklearn.__version__}")
check("Python version is 3.10 - 3.13", (3, 10) <= sys.version_info[:2] <= (3, 13),
      f"found {sys.version.split()[0]}")

# ----------------------------------------------------------------------
print("\n== 2. Loading (any sklearn version warning counts as a failure) ==")
from deal_advisor import AdvisorInputError, DealAdvisor, get_deal_rating   # noqa: E402

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    adv = DealAdvisor()
bad = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
check("model + preprocessor load without InconsistentVersionWarning", not bad, f"{len(bad)} warning(s)")
check("final model is Random Forest", adv.model_name == "Random Forest", adv.model_name)

# ----------------------------------------------------------------------
print("\n== 3. The saved files belong together (Train rows, batch vs one-car path) ==")
train = adv.train
X = train[adv.feature_columns]
batch = adv.model.predict(adv.preprocessor.transform(X))
y = train["price"].to_numpy(float)
r2 = 1 - ((y - batch) ** 2).sum() / ((y - y.mean()) ** 2).sum()
with open("part12_final_model.json", encoding="utf-8") as f:
    expected_r2 = json.load(f)["train_r2"]
check("Train R2 matches the value saved in Part 12", abs(r2 - expected_r2) < 0.002,
      f"now {r2:.4f} vs saved {expected_r2:.4f}")

sample = train[(train["engineSize"].notna()) & (train["engineSize"] > 0)].sample(300, random_state=0)
max_gap, skipped = 0.0, 0
for idx, row in sample.iterrows():
    car = {"make": row["Make"], "model": row["model"], "year": int(row["year"]), "mileage": float(row["mileage"]),
           "transmission": row["transmission"], "fuel_type": row["fuelType"],
           "engine_size": float(row["engineSize"]), "mpg": float(row["mpg"]), "tax": float(row["tax"])}
    try:
        one_car = adv.estimate(car)["predicted_price"]
    except AdvisorInputError:
        skipped += 1
        continue
    max_gap = max(max_gap, abs(one_car - round(float(batch[train.index.get_loc(idx)]))))
check("one-car feature building == Part 6 features (300 random Train cars)", max_gap <= 1,
      f"largest gap £{max_gap:.0f}, {skipped} rows skipped")

# ----------------------------------------------------------------------
print("\n== 4. Deal rating rules (market £10,000; the % is rounded to 1 decimal BEFORE rating) ==")
rating_cases = [(8900, "great"), (9000, "good"), (9500, "good"), (9501, "good"), (9600, "fair"),
                (10000, "fair"), (10449, "fair"), (10499, "slightly"), (10500, "slightly"), (11000, "slightly"),
                (11001, "slightly"), (11100, "over"), (15000, "over")]
for seller, want in rating_cases:
    got = get_deal_rating(10000, seller)
    check(f"seller £{seller:,} -> {want}", got["deal_key"] == want, f"{got['below_market_pct']}% -> {got['deal_key']}")
check("invalid prices are rejected",
      all(raises(lambda v=v: get_deal_rating(10000, v), ValueError) for v in [0, -5, float("nan"), "abc", None, True]))

# ----------------------------------------------------------------------
print("\n== 5. Input validation ==")
ok_car = {"make": "Ford", "model": "Focus", "year": 2017, "mileage": 30000, "transmission": "Manual",
          "fuel_type": "Petrol", "engine_size": 1.0}
check("a valid car is accepted", adv.estimate(ok_car)["predicted_price"] > 0)
bad_cars = {
    "unknown make": {**ok_car, "make": "Tesla"},
    "model of another make": {**ok_car, "model": "A1"},
    "year after the data (2025)": {**ok_car, "year": 2025},
    "year before 1980": {**ok_car, "year": 1970},
    "negative mileage": {**ok_car, "mileage": -1},
    "engine size 0 for a petrol car": {**ok_car, "engine_size": 0},
    "missing engine size (petrol)": {k: v for k, v in ok_car.items() if k != "engine_size"},
    "mileage is text": {**ok_car, "mileage": "lots"},
}
for name, car in bad_cars.items():
    check(f"rejected: {name}", raises(lambda c=car: adv.estimate(c), AdvisorInputError))
check("rejected: description is not a dict", raises(lambda: adv.estimate("a car"), AdvisorInputError))

# ----------------------------------------------------------------------
print("\n== 6. Common-sense behaviour (Ford Focus, Manual, Petrol, 1.0L) ==")


def price(**kw):
    return adv.estimate({**ok_car, **kw})["predicted_price"]


check("more mileage -> lower price", price(mileage=20000) > price(mileage=60000) > price(mileage=120000),
      f"{price(mileage=20000):,.0f} > {price(mileage=60000):,.0f} > {price(mileage=120000):,.0f}")
check("newer car -> higher price", price(year=2019) > price(year=2016) > price(year=2012),
      f"{price(year=2019):,.0f} > {price(year=2016):,.0f} > {price(year=2012):,.0f}")
bmw = adv.estimate({"make": "BMW", "model": "3 Series", "year": 2017, "mileage": 30000,
                    "transmission": "Manual", "fuel_type": "Diesel", "engine_size": 2.0})["predicted_price"]
check("a BMW 3 Series costs more than a Ford Focus of the same year/mileage", bmw > price(),
      f"BMW {bmw:,.0f} vs Focus {price():,.0f}")
res = adv.advise(ok_car, price() * 0.9)
check("advise(): asking price 10% under the estimate -> a 'good' or 'great' deal",
      res["deal_key"] in ("good", "great"), f"{res['deal_rating']} ({res['below_market_pct']}%)")
check("advise(): an asking price 4x the estimate triggers the typo warning",
      len(adv.advise(ok_car, price() * 4)["warnings"]) == 1)
check("missing MPG and tax are filled in and reported", len(adv.estimate(ok_car)["imputed"]) == 2)

# ----------------------------------------------------------------------
passed = sum(results)
print(f"\n{'=' * 60}\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
