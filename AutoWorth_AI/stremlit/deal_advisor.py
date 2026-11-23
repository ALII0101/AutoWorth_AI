# =====================================================================
# deal_advisor.py  --  the decision layer of AutoWorth AI
# =====================================================================
# Used by Part 14 (tests + examples) and, later, by the Streamlit app (Part 15):
#       from deal_advisor import DealAdvisor, format_report
#       adv = DealAdvisor()
#       print(format_report(adv.advise({...car...}, seller_price)))
#
# Files it needs in the SAME folder (all created by earlier parts):
#   part12_final_model.json   -> which model is the final one + its file     (Part 12)
#   models/<final>.joblib     -> the final model                              (Part 8 / 11)
#   preprocessor.joblib       -> the fitted ColumnTransformer                 (Part 7)
#   train_fe.csv              -> Train rows with the Part 6 features          (Part 6)
#
# What it does NOT do: no training, no tuning, no Test data. It rebuilds the Part 6 features and the
# Part 7 matrix for ONE car, asks the Part 12 model for a price, and compares it with the asking price.

import json
import math
import numbers
import os
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants (fixed here, not tuned on any data)
# ----------------------------------------------------------------------
DECISION_FILE = "part12_final_model.json"
PREPROCESSOR_FILE = "preprocessor.joblib"
TRAIN_FILE = "train_fe.csv"

# must match Part 6 / Part 7 -- DealAdvisor() checks both against train_fe.csv when it starts
AGE_REFERENCE_YEAR = 2020
PREMIUM_MAKES = {"Mercedes", "Audi", "BMW"}
CATEGORICAL = ["Make", "model", "transmission", "fuelType"]
MIN_VALID_YEAR = 1980                 # Part 3 removed year < 1980 rows

# Deal rating, in percent below the estimated market price: (market - seller) / market x 100.
# The assignment does not say which side the exact boundaries belong to; this rule is symmetric:
#   > 10 Great | 5..10 Good | between -5 and 5 Fair | -10..-5 Slightly Overpriced | < -10 Overpriced
GREAT_ABOVE = 10.0
GOOD_FROM = 5.0
FAIR_ABOVE = -5.0
SLIGHTLY_FROM = -10.0
RATING_LABELS = {                      # insertion order = best deal first
    "great": ("Great Deal", "🟢"),
    "good": ("Good Deal", "🟢"),
    "fair": ("Fair Price", "🟡"),
    "slightly": ("Slightly Overpriced", "🟠"),
    "over": ("Overpriced", "🔴"),
}

# Reliability notes (operational warning levels, NOT statistical limits)
LOW_PRICE_WARN = 5_000                # estimates below this: errors are relatively larger (Part 13)
HIGH_PRICE_WARN = 40_000              # estimates at/above this: few training cars, larger errors (Part 13)
RARE_MODEL_N = 30                     # fewer Train cars than this for the Make+Model -> note
ENGINE_REQUIRED_SHARE = 0.95          # if >= 95% of Train cars of a fuel type have an engine size, it is required
MIN_SIMILAR_CARS = 5                  # a "similar cars" group needs at least this many rows for a median
PRICE_TYPO_FACTOR = 3.0               # asking price > 3x or < 1/3 of the estimate -> "check for a typo"


class AdvisorInputError(ValueError):
    """The car description cannot be valued. `.errors` is a list of plain-English messages."""

    def __init__(self, errors):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


# ----------------------------------------------------------------------
# Deal rating (pure business rule: no model, no data)
# ----------------------------------------------------------------------
def _valid_price(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Real, np.number)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive, finite number, got {value!r}")
    return v


def get_deal_rating(predicted_price, seller_price):
    """Compare the asking price with the estimated market price. Raises ValueError for invalid prices."""
    market = _valid_price(predicted_price, "estimated market price")
    seller = _valid_price(seller_price, "seller price")

    diff = market - seller                                    # > 0 means the seller is cheaper
    pct = round(diff / market * 100, 1)                       # rounded FIRST: 4.96% is shown (and rated) as 5.0%
    if pct > GREAT_ABOVE:
        key = "great"
    elif pct >= GOOD_FROM:
        key = "good"
    elif pct > FAIR_ABOVE:
        key = "fair"
    elif pct >= SLIGHTLY_FROM:
        key = "slightly"
    else:
        key = "over"
    label, icon = RATING_LABELS[key]

    whole = round(diff)
    if whole == 0:
        text = "same as the estimated market price"
    elif diff > 0:
        text = f"£{whole:,} cheaper"
    else:
        text = f"£{abs(whole):,} more expensive"
    return {
        "predicted_price": market, "seller_price": seller, "difference_gbp": float(diff),
        "below_market_pct": float(pct), "deal_key": key, "deal_rating": label, "deal_icon": icon,
        "difference_text": text,
    }


# ----------------------------------------------------------------------
# The advisor
# ----------------------------------------------------------------------
class DealAdvisor:
    def __init__(self, decision_file=None, preprocessor_file=None, train_file=None):
        decision_file = decision_file or DECISION_FILE
        preprocessor_file = preprocessor_file or PREPROCESSOR_FILE
        train_file = train_file or TRAIN_FILE
        for path, made_in in [(decision_file, "Part 12"), (preprocessor_file, "Part 7"), (train_file, "Part 6")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found in {os.getcwd()} (it is created in {made_in}). "
                                        "Run that part first, or copy the file into this folder.")

        with open(decision_file, encoding="utf-8") as f:
            self.decision = json.load(f)
        self.model_name = self.decision["final_model"]
        self.model_file = self.decision["model_file"]
        if not os.path.exists(self.model_file):
            # the JSON says models/<name>.joblib; also accept the model placed directly in this folder
            flat = os.path.basename(self.model_file)
            if os.path.exists(flat):
                self.model_file = flat
            else:
                raise FileNotFoundError(f"{self.model_file} (the final model named in {decision_file}) not found "
                                        f"in {os.getcwd()}. Put it in a 'models' folder or next to this file.")
        self.model = joblib.load(self.model_file)
        self.preprocessor = joblib.load(preprocessor_file)
        self.feature_columns = list(self.preprocessor.feature_names_in_)

        train = pd.read_csv(train_file)
        for c in CATEGORICAL:                                  # same deterministic cleanup as Part 7 (7.2)
            train[c] = train[c].astype(str).str.strip()
        self.train = train
        self._verify_constants()

        self.pair_counts = train.groupby(["Make", "model"]).size()
        self._pairs = set(self.pair_counts.index)
        self._build_lookups()

    # ---- start-up checks / lookups ----------------------------------
    def _verify_constants(self):
        t = self.train
        if not (t["year"] + t["car_age"] == AGE_REFERENCE_YEAR).all():
            raise RuntimeError(f"AGE_REFERENCE_YEAR = {AGE_REFERENCE_YEAR} does not match car_age in train_fe.csv")
        if not (t["is_premium_brand"] == t["Make"].isin(PREMIUM_MAKES).astype(int)).all():
            raise RuntimeError("PREMIUM_MAKES does not match is_premium_brand in train_fe.csv")
        expected = np.where(t["car_age"] > 0, t["mileage"] / t["car_age"].where(t["car_age"] > 0, np.nan), np.nan)
        if not np.allclose(expected, t["mileage_per_year"].to_numpy(float), equal_nan=True):
            raise RuntimeError("mileage_per_year in train_fe.csv does not follow mileage / car_age")

    def _build_lookups(self):
        t = self.train
        self._cats, self._lower = {}, {}
        for col in ["Make", "transmission", "fuelType"]:
            counts = t[col].value_counts()
            self._cats[col] = set(counts.index)
            lower = {}
            for v in counts.index:                              # most frequent spelling wins a case collision
                lower.setdefault(v.lower(), v)
            self._lower[col] = lower
        self._models_by_make = {}
        for make, grp in t.groupby("Make"):
            counts = grp["model"].value_counts()
            lower = {}
            for v in counts.index:
                lower.setdefault(v.lower(), v)
            self._models_by_make[make] = (set(counts.index), lower)
        self.ranges = {k: (float(t[c].min()), float(t[c].max()))
                       for k, c in [("year", "year"), ("mileage", "mileage"), ("engine_size", "engineSize"),
                                    ("mpg", "mpg"), ("tax", "tax")]}
        self._engine_present = t.groupby("fuelType")["engineSize"].apply(lambda s: s.notna().mean())
        self._engine_zero = t.groupby("fuelType")["engineSize"].apply(lambda s: (s == 0).mean())

    # ---- small helpers ----------------------------------------------
    @staticmethod
    def _is_missing(v):
        return v is None or (isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isnan(float(v)))

    def _number(self, value, label, errors, required=False):
        """Return a finite float, or None when the value is missing (an error is recorded if required)."""
        if self._is_missing(value):
            if required:
                errors.append(f"{label} is required")
            return None
        if isinstance(value, (bool, np.bool_)):
            errors.append(f"{label} must be a number, got {value!r}")
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number, got {value!r}")
            return None
        if not math.isfinite(v):
            errors.append(f"{label} must be a finite number, got {value!r}")
            return None
        return v

    def _match(self, col, value, label, errors):
        if self._is_missing(value) or not isinstance(value, str) or not value.strip():
            errors.append(f"{label} is required")
            return None
        s = value.strip()
        if s in self._cats[col]:
            return s
        hit = self._lower[col].get(s.lower())
        if hit is None:
            known = ", ".join(sorted(self._cats[col]))
            errors.append(f"unknown {label} '{value.strip()}'. Known values: {known}")
        return hit

    def _match_model(self, make, value, errors):
        if self._is_missing(value) or not isinstance(value, str) or not value.strip():
            errors.append("model is required")
            return None
        s = value.strip()
        exact, lower = self._models_by_make[make]
        if s in exact:
            return s
        hit = lower.get(s.lower())
        if hit is not None:
            return hit
        owners = sorted({mk for mk in self._models_by_make if s.lower() in self._models_by_make[mk][1]})
        if owners:
            errors.append(f"'{s}' is not a {make} model in the training data (it belongs to: {', '.join(owners)})")
        else:
            some = ", ".join(sorted(exact)[:15])
            errors.append(f"unknown {make} model '{s}'. Examples of known {make} models: {some}")
        return None

    def _similar_median(self, column, make, model, fuel):
        """Median of a Train column over similar cars (Make+Model+Fuel, then Make+Model, then Fuel)."""
        t = self.train
        groups = [("same make, model and fuel type", (t["Make"] == make) & (t["model"] == model) & (t["fuelType"] == fuel)),
                  ("same make and model", (t["Make"] == make) & (t["model"] == model)),
                  ("same fuel type", t["fuelType"] == fuel)]
        for name, mask in groups:
            vals = t.loc[mask, column].dropna()
            if len(vals) >= MIN_SIMILAR_CARS:
                return float(vals.median()), name
        return None, None

    # ---- feature building (Part 6) + model call (Part 7 / 12) --------
    def assemble_features(self, clean):
        """One-row DataFrame with the Part 6 features, in the columns the Part 7 preprocessor was fitted on.

        `clean` uses the Train column names: Make, model, year, mileage, transmission, fuelType, tax, mpg,
        engineSize (engineSize may be NaN/None)."""
        year = float(clean["year"])
        mileage = float(clean["mileage"])
        car_age = AGE_REFERENCE_YEAR - year
        row = {
            "Make": clean["Make"], "model": clean["model"], "transmission": clean["transmission"],
            "fuelType": clean["fuelType"], "year": year, "mileage": mileage,
            "tax": float(clean["tax"]), "mpg": float(clean["mpg"]),
            "engineSize": np.nan if self._is_missing(clean.get("engineSize")) else float(clean["engineSize"]),
            "car_age": car_age,
            "mileage_per_year": mileage / car_age if car_age > 0 else np.nan,   # same rule as Part 6
            "is_premium_brand": int(clean["Make"] in PREMIUM_MAKES),
        }
        return pd.DataFrame([row])[self.feature_columns]

    def predict_price(self, feats):
        processed = self.preprocessor.transform(feats)
        return float(self.model.predict(processed)[0])

    # ---- public API --------------------------------------------------
    def estimate(self, raw):
        """Validate a car description and estimate its market price. Raises AdvisorInputError."""
        if not isinstance(raw, Mapping):
            raise AdvisorInputError("the car description must be a dict")
        errors, notes_imputed, reasons = [], [], []

        make = self._match("Make", raw.get("make"), "make", errors)
        model = self._match_model(make, raw.get("model"), errors) if make else None
        trans = self._match("transmission", raw.get("transmission"), "transmission", errors)
        fuel = self._match("fuelType", raw.get("fuel_type"), "fuel type", errors)

        year = self._number(raw.get("year"), "year", errors, required=True)
        if year is not None:
            if not year.is_integer():
                errors.append(f"year must be a whole number (e.g. 2017), got {raw.get('year')!r}")
            elif year > AGE_REFERENCE_YEAR:
                errors.append(f"year {int(year)} is newer than the data (latest model year is "
                              f"{AGE_REFERENCE_YEAR}); the advisor cannot value it")
            elif year < MIN_VALID_YEAR:
                errors.append(f"year {int(year)} is before {MIN_VALID_YEAR}, which the data does not cover")
        mileage = self._number(raw.get("mileage"), "mileage", errors, required=True)
        if mileage is not None and mileage < 0:
            errors.append(f"mileage cannot be negative, got {raw.get('mileage')!r}")
            mileage = None
        engine = self._number(raw.get("engine_size"), "engine size", errors)
        mpg = self._number(raw.get("mpg"), "mpg", errors)
        tax = self._number(raw.get("tax"), "tax", errors)
        if mpg is not None and mpg <= 0:
            errors.append(f"mpg must be above 0, got {raw.get('mpg')!r}")
            mpg = None
        if tax is not None and tax < 0:
            errors.append(f"tax cannot be negative, got {raw.get('tax')!r}")
            tax = None
        if engine is not None and fuel is not None:
            if engine < 0 or (engine == 0 and self._engine_zero.get(fuel, 0.0) < 0.5):
                errors.append(f"engine size {raw.get('engine_size')!r} L is not valid for {fuel} cars")
                engine = None
        if engine is None and fuel is not None and self._is_missing(raw.get("engine_size")):
            if self._engine_present.get(fuel, 0.0) >= ENGINE_REQUIRED_SHARE:
                errors.append(f"engine size is required for {fuel} cars "
                              f"({self._engine_present[fuel]:.0%} of the training cars have one)")
        if errors:
            raise AdvisorInputError(errors)

        # ---- fill empty optional fields with Train medians of similar cars (and say so) ----
        values = {"engine_size": engine, "mpg": mpg, "tax": tax}
        columns = {"engine_size": "engineSize", "mpg": "mpg", "tax": "tax"}
        labels = {"engine_size": "engine size", "mpg": "MPG", "tax": "tax"}
        for k in ["engine_size", "mpg", "tax"]:
            if values[k] is None:
                med, group = self._similar_median(columns[k], make, model, fuel)
                if med is None:
                    notes_imputed.append(f"{labels[k]} was not given: the model's own imputer fills in the "
                                         "Train median")
                else:
                    values[k] = med
                    notes_imputed.append(f"{labels[k]} was not given: filled with {med:g}, the Train median of "
                                         f"cars with the {group}")

        clean = {"Make": make, "model": model, "year": year, "mileage": mileage, "transmission": trans,
                 "fuelType": fuel, "tax": values["tax"] if values["tax"] is not None else np.nan,
                 "mpg": values["mpg"] if values["mpg"] is not None else np.nan,
                 "engineSize": values["engine_size"]}
        feats = self.assemble_features(clean)
        price = self.predict_price(feats)
        if not math.isfinite(price) or price <= 0:
            raise AdvisorInputError("the model returned an unusable estimate for this car")
        price = float(round(price))

        # ---- reliability notes (independent of the rating) ----
        n_pair = int(self.pair_counts.get((make, model), 0))
        if price < LOW_PRICE_WARN:
            reasons.append(f"estimate below £{LOW_PRICE_WARN:,}: errors are relatively larger for cheap cars")
        if price >= HIGH_PRICE_WARN:
            reasons.append(f"estimate of £{HIGH_PRICE_WARN:,} or more: few such cars in the training data and "
                           "errors are larger")
        if n_pair < RARE_MODEL_N:
            reasons.append(f"only {n_pair} training cars of this make and model (fewer than {RARE_MODEL_N})")
        if notes_imputed:
            reasons.append("some details were filled in, not given")
        given = {"year": year, "mileage": mileage, "engine_size": engine, "mpg": mpg, "tax": tax}
        for k, v in given.items():
            lo, hi = self.ranges[k]
            if v is not None and (v < lo or v > hi):
                reasons.append(f"{k.replace('_', ' ')} {v:,g} is outside the training range ({lo:,g} to {hi:,g})")

        car = {"make": make, "model": model, "year": int(year), "mileage": mileage, "transmission": trans,
               "fuel_type": fuel, "engine_size": values["engine_size"], "mpg": values["mpg"], "tax": values["tax"]}
        return {"predicted_price": price, "car": car, "imputed": notes_imputed,
                "reliability": "Check carefully" if reasons else "Standard",
                "reliability_reasons": reasons, "pair_count": n_pair}

    def advise(self, raw, seller_price):
        """Estimate the market price and rate the asking price. Raises AdvisorInputError / ValueError."""
        seller = _valid_price(seller_price, "seller price")
        est = self.estimate(raw)
        rating = get_deal_rating(est["predicted_price"], seller)
        warnings = []
        if seller > PRICE_TYPO_FACTOR * est["predicted_price"] or seller < est["predicted_price"] / PRICE_TYPO_FACTOR:
            warnings.append(f"the asking price (£{seller:,.0f}) is very far from the estimate "
                            f"(£{est['predicted_price']:,.0f}): check it for a typo")
        return {**est, **rating, "warnings": warnings}


# ----------------------------------------------------------------------
# Text report (same layout the Streamlit app will show)
# ----------------------------------------------------------------------
def format_report(res):
    car = res["car"]
    pct = res["below_market_pct"]
    if pct > 0:
        pct_text = f" ({pct:.1f}% below the estimate)"
    elif pct < 0:
        pct_text = f" ({abs(pct):.1f}% above the estimate)"
    else:
        pct_text = ""
    eng = car["engine_size"]
    eng_txt = "engine size n/a" if eng is None or math.isnan(float(eng)) else f"{eng:g}L"
    desc = (f"{car['year']} {car['make']} {car['model']} | {car['transmission']} | {car['fuel_type']} | "
            f"{eng_txt} | {car['mileage']:,.0f} miles")
    lines = [
        f"   Car                    : {desc}",
        f"   💰 Estimated Market Price: £{res['predicted_price']:,.0f}",
        f"   🏷️ Seller Price          : £{res['seller_price']:,.0f}",
        f"   💵 Difference            : {res['difference_text']}{pct_text}",
        f"   🚦 Deal Rating           : {res['deal_icon']} {res['deal_rating'].upper()}",
        f"   Estimate reliability   : {res['reliability']}",
    ]
    for r in res["reliability_reasons"]:
        lines.append(f"      - {r}")
    for n in res["imputed"]:
        lines.append(f"   Note: {n}")
    for w in res["warnings"]:
        lines.append(f"   Warning: {w}")
    return "\n".join(lines)


if __name__ == "__main__":                       # quick self-check:  python deal_advisor.py
    adv = DealAdvisor()
    pair = adv.pair_counts.idxmax()
    row = adv.train[(adv.train["Make"] == pair[0]) & (adv.train["model"] == pair[1])
                    & adv.train["engineSize"].notna()].iloc[0]
    car = {"make": row["Make"], "model": row["model"], "year": int(row["year"]), "mileage": float(row["mileage"]),
           "transmission": row["transmission"], "fuel_type": row["fuelType"],
           "engine_size": float(row["engineSize"]), "mpg": float(row["mpg"]), "tax": float(row["tax"])}
    est = adv.estimate(car)["predicted_price"]
    print(f"Final model: {adv.model_name}")
    print(format_report(adv.advise(car, round(est * 0.93))))
