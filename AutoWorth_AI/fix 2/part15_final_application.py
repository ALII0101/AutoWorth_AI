# =====================================================================
# PART 15: Final Application
# =====================================================================
# Input : deal_advisor.py, and the same files Part 14 already required
#         (part12_final_model.json, preprocessor.joblib, train_fe.csv, models/*.joblib)
# Output: this script's log. The user-facing app is app.py
#         (streamlit run app.py)
#
# Rules:
#   1. No training, no tuning, no Test-set predictions, no new rating rules.
#   2. The Streamlit page is a UI over DealAdvisor.advise(...). This script
#      checks that the advisor — the same object the app loads — still behaves
#      as the application needs, without re-implementing Part 14.
#   3. App.py must not compute predicted_price, below_market_pct, or ratings.

import os
from pathlib import Path

try:
    os.chdir(Path(__file__).resolve().parent)   # when run as a .py file
except NameError:
    pass                                         # in Colab/Jupyter __file__ does not exist: stay in the current folder

from deal_advisor import AGE_REFERENCE_YEAR, AdvisorInputError, DealAdvisor


def banner(title):
    print("=" * 90)
    print(title)
    print("=" * 90)


def expect_ok(label, fn, check):
    try:
        res = fn()
    except Exception as exc:
        raise AssertionError(f"{label}: expected a result, got {type(exc).__name__}: {exc}") from exc
    msg = check(res)
    if msg:
        raise AssertionError(f"{label}: {msg}")
    print(f"   OK  {label}")
    return res


def expect_error(label, fn, exc_type):
    try:
        fn()
    except exc_type as exc:
        detail = exc.errors[0] if isinstance(exc, AdvisorInputError) else str(exc)
        print(f"   OK  {label}  ->  {detail[:110]}")
        return
    raise AssertionError(f"{label}: invalid input was accepted")


banner("15.1 FINAL APPLICATION")
print("Part 15 is a simple Streamlit page (app.py). It does not train a model.")
print("Flow: user input -> DealAdvisor.advise(car, seller_price) -> display the returned dict.")
print("Run the app with:  streamlit run app.py")
print()

banner("15.2 LOAD THE SAME ADVISOR THE APP USES")
adv = DealAdvisor()
print(f"Final model: {adv.model_name}")
print(f"Latest model year the UI may offer: {AGE_REFERENCE_YEAR}")
print(f"Makes available to the Make dropdown: {sorted(adv.train['Make'].unique())}")
print()

makes = list(adv.train["Make"].value_counts().index)
transmissions = list(adv.train["transmission"].value_counts().index)
fuels = list(adv.train["fuelType"].value_counts().index)
assert makes and transmissions and fuels, "training data has empty dropdowns"
bmw_models = [str(m) for m in adv.pair_counts.xs("BMW", level="Make").index] if "BMW" in makes else []
assert "3 Series" in bmw_models, "the app's BMW / 3 Series default is not in Train"

base = {
    "make": "BMW",
    "model": "3 Series",
    "year": 2019,
    "mileage": 35000,
    "transmission": "Automatic" if "Automatic" in transmissions else transmissions[0],
    "fuel_type": "Petrol" if "Petrol" in fuels else fuels[0],
    "engine_size": 2.0,
    "mpg": 45.0,
    "tax": 150.0,
}

banner("15.3 APPLICATION ACCEPTANCE TESTS  (UI contract over DealAdvisor)")
print("These tests call advise() / estimate() the same way app.py does. They do not score Test.")
print()

est = expect_ok(
    "Valid BMW 3 Series returns a price",
    lambda: adv.estimate(base),
    lambda r: None if r["predicted_price"] > 0 else "missing predicted_price",
)
market = est["predicted_price"]
print(f"       estimated market price: £{market:,.0f}")

expect_ok(
    "Seller 20% below -> Great Deal",
    lambda: adv.advise(base, round(market * 0.80)),
    lambda r: None if r["deal_rating"] == "Great Deal" else f"got {r['deal_rating']}",
)
expect_ok(
    "Seller 10% below -> Good Deal",
    lambda: adv.advise(base, round(market * 0.90)),
    lambda r: None if r["deal_rating"] == "Good Deal" else f"got {r['deal_rating']}",
)
expect_ok(
    "Seller 5% below -> Good Deal",
    lambda: adv.advise(base, round(market * 0.95)),
    lambda r: None if r["deal_rating"] == "Good Deal" else f"got {r['deal_rating']}",
)
expect_ok(
    "Equal price -> Fair Price",
    lambda: adv.advise(base, market),
    lambda r: None if r["deal_rating"] == "Fair Price" else f"got {r['deal_rating']}",
)
expect_ok(
    "Seller 5% above -> Slightly Overpriced",
    lambda: adv.advise(base, round(market * 1.05)),
    lambda r: None if r["deal_rating"] == "Slightly Overpriced" else f"got {r['deal_rating']}",
)
expect_ok(
    "Seller 10% above -> Slightly Overpriced",
    lambda: adv.advise(base, round(market * 1.10)),
    lambda r: None if r["deal_rating"] == "Slightly Overpriced" else f"got {r['deal_rating']}",
)
expect_ok(
    "Seller more than 10% above -> Overpriced",
    lambda: adv.advise(base, round(market * 1.20)),
    lambda r: None if r["deal_rating"] == "Overpriced" else f"got {r['deal_rating']}",
)

expect_error("Negative mileage -> error", lambda: adv.estimate({**base, "mileage": -1000}), AdvisorInputError)
expect_error(
    "Unknown Make+Model combination -> error",
    lambda: adv.estimate({**base, "make": "BMW", "model": "Fiesta"}),
    AdvisorInputError,
)
expect_error("Year after 2020 -> error", lambda: adv.estimate({**base, "year": 2024}), AdvisorInputError)
expect_error("Seller price 0 -> error", lambda: adv.advise(base, 0), ValueError)
expect_error("Seller price negative -> error", lambda: adv.advise(base, -5000), ValueError)

no_mpg_tax = {k: v for k, v in base.items() if k not in ("mpg", "tax")}
imputed = expect_ok(
    "Empty MPG/Tax -> estimate plus imputation notes",
    lambda: adv.advise(no_mpg_tax, round(market * 0.93)),
    lambda r: None if (r["predicted_price"] > 0 and r["imputed"]) else "expected imputed notes and a price",
)
print(f"       reliability: {imputed['reliability']}")
for note in imputed["imputed"]:
    print(f"       note: {note[:120]}")

expensive = expect_ok(
    "Expensive / rare car -> reliability Check carefully",
    lambda: adv.advise(
        {
            "make": "Mercedes",
            "model": "G Class",
            "year": 2019,
            "mileage": 12000,
            "transmission": "Automatic",
            "fuel_type": "Petrol",
            "engine_size": 4.0,
            "mpg": 22.0,
            "tax": 580,
        },
        120000,
    ),
    lambda r: None if r["reliability"] == "Check carefully" else f"got {r['reliability']}",
)
print(f"       estimated £{expensive['predicted_price']:,.0f}  |  {expensive['deal_icon']} {expensive['deal_rating']}")
print()

banner("15.4 KEYS THE STREAMLIT PAGE MAY DISPLAY")
required = [
    "predicted_price", "seller_price", "difference_text", "below_market_pct",
    "deal_rating", "deal_icon", "deal_key", "reliability", "reliability_reasons",
    "imputed", "warnings", "car",
]
missing = [k for k in required if k not in imputed]
assert not missing, f"advisor result is missing keys the UI needs: {missing}"
print("OK: advise() still returns every field app.py displays.")
print("    app.py must not recompute rating, below_market_pct, or reliability.")
print()

banner("PART 15 LOG")
print("UI file: app.py   |   logic file: deal_advisor.py (unchanged)")
print("No model was trained or changed. Test data is not used in the application.")
print("Start the page with:  streamlit run app.py")
print("Required artefacts next to app.py: part12_final_model.json, preprocessor.joblib,")
print("train_fe.csv, and the final model file named in part12_final_model.json.")
