# =====================================================================
# PART 13: Understand Your Model  (descriptive analysis of the FINAL model)
# =====================================================================
# Input : part12_final_model.json, test_predictions.csv (Part 12),
#         models/*.joblib, preprocessed_data.joblib + feature_names.csv (Part 7)
#         optional: SPLIT_FILES = the per-split CSVs with the ORIGINAL (unencoded / unscaled) columns, see CONFIG
# Output: part13_actual_vs_predicted.png, part13_residuals.png, part13_feature_importance.png,
#         part13_relationships.png, part13_feature_importance.csv, part13_grouped_importance.csv,
#         part13_relationship_bins.csv, part13_largest_abs_errors.csv, part13_largest_pct_errors.csv,
#         part13_error_by_band.csv
#
# Rules:
#   1. NOTHING is trained or tuned here. The model is loaded from disk and only used to predict.
#   2. The final model comes from part12_final_model.json. It is never re-selected.
#   3. Test is used only to DESCRIBE the final model (plots, errors). Feature importance uses
#      Validation, so no Test-based number can influence a modelling decision.
#   4. Everything here is an association in this dataset, not a cause.
#      If a problem is found, write it as a finding. If you want to improve the model later,
#      iterate on Validation, not on Test.

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RANDOM_STATE = 42

DATA_FILE = "preprocessed_data.joblib"
FEATURES_FILE = "feature_names.csv"
DECISION_FILE = "part12_final_model.json"
PRED_FILE = "test_predictions.csv"

# Original columns (Make, Model, Year, Mileage, Engine size ...) are needed for readable tables.
#  - If your preprocessed numeric columns are NOT scaled, they are read straight from X.
#  - If they ARE scaled (or you want the original Make/Model text), SPLIT_FILES points to the CSV of EACH split
#    (the same files Part 7 read: one row per car, same row ORDER as X_train / X_val / X_test).
#    Rows are matched by POSITION and every split is checked against y (price) before it is used, so a file
#    from the wrong split, or a re-ordered one, stops the script instead of giving wrong tables.
#    Set SPLIT_FILES = None to rebuild everything from X only (no original Make/Model text).
SPLIT_FILES = {"Train": "train_fe.csv", "Validation": "val_fe.csv", "Test": "test_fe.csv"}
REFERENCE_YEAR = None        # only used if the data has Year but no Car Age column, e.g. 2020

# One-hot column name prefixes -> original feature. Names are matched case-insensitively after
# removing sklearn prefixes such as "cat__". Add your own prefix if a group is not detected.
CATEGORICAL_GROUPS = {
    "Make": ["make", "manufacturer", "brand"],
    "Model": ["model"],
    "Transmission": ["transmission"],
    "Fuel type": ["fueltype", "fuel_type", "fuel type", "fuel"],
}
# Numeric columns that carry (almost) the same information are permuted TOGETHER, otherwise each of
# them looks unimportant because the others still hold the information. Names are matched after
# lower-casing and removing non-alphanumerics ("Car_Age" -> "carage").
MERGE_GROUPS = {
    "Age / Year": ["year", "carage", "age", "vehicleage"],
}
NUMERIC_ALIASES = {            # original numeric columns used in the tables / plots
    "Year": ["year"],
    "Age": ["carage", "age", "vehicleage"],
    "Mileage": ["mileage"],
    "Engine size": ["enginesize"],
}
# Engineered features are derived from each other (mileage_per_year = mileage / car_age), so shuffling ONE of
# them leaves the information in the others and understates its importance. FAMILIES shuffle several groups
# together to show how much the model relies on that kind of information as a whole (names = group names
# printed in the 13.0 table; missing names are ignored).
FAMILIES = {
    "Usage: car_age + mileage + mileage_per_year": ["car_age", "mileage", "mileage_per_year", "Age / Year"],
    "Brand: Make + Model + is_premium_brand": ["Make", "Model", "is_premium_brand"],
}
RAW_CATEGORICAL_ALIASES = {"Make": ["make", "manufacturer", "brand"], "Model": ["model"]}

N_PERM_ROWS = 5000           # Validation rows used for permutation importance (speed)
N_PERM_REPEATS = 5
TOP_N_ERRORS = 10
MIN_N = 30                   # groups smaller than this are flagged / hidden
ZOOM_MAX = 40_000            # right-hand Actual vs Predicted panel zooms to this price

PRICE_EDGES = [0, 5_000, 10_000, 20_000, 40_000, 75_000, np.inf]
MILEAGE_EDGES = [0, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, np.inf]
AGE_EDGES = [0, 2, 4, 6, 8, 10, 12, np.inf]
ENGINE_EDGES = [0, 1.1, 1.6, 2.1, 3.1, np.inf]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def banner(title):
    print("=" * 90)
    print(title)
    print("=" * 90)


def gbp(v):
    return f"£{v:,.0f}"


def strip_prefix(name):
    return re.sub(r"^(num|cat|cont|bin|binary|ord|remainder|numeric|categorical|continuous)__", "", str(name))


def norm(name):
    return re.sub(r"[^a-z0-9]", "", strip_prefix(name).lower())


def make_labels(edges, fmt):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{fmt(lo)}+" if np.isinf(hi) else f"{fmt(lo)}–{fmt(hi)}")
    return out


def f_money(v):
    return f"£{v / 1000:g}k"


def f_thousand(v):
    return f"{v / 1000:g}k"


def f_plain(v):
    return f"{v:g}"


def group_of(feature):
    """Return (group name, one-hot label or None) for a preprocessed column."""
    s = strip_prefix(feature)
    for g, prefixes in CATEGORICAL_GROUPS.items():
        for p in prefixes:
            m = re.match(rf"^{re.escape(p)}[_\s=:\-]+(.+)$", s, flags=re.IGNORECASE)
            if m:
                return g, m.group(1)
    n = norm(feature)
    for g, names in MERGE_GROUPS.items():
        if n in names:
            return g, None
    return s, None


# ----------------------------------------------------------------------
# 13.0 Load everything and prove it matches Part 12
# ----------------------------------------------------------------------
banner("13.0 SETUP AND INTEGRITY CHECKS  (nothing is trained in this part)")
for f in [DECISION_FILE, PRED_FILE, DATA_FILE, FEATURES_FILE]:
    assert os.path.exists(f), f"missing file: {f} (run Part 12 first)"

decision = json.load(open(DECISION_FILE))
assert decision.get("test") is not None, "Part 12 has not evaluated Test yet"
final_name = decision["final_model"]
final_model = joblib.load(decision["model_file"])
if hasattr(final_model, "n_jobs"):
    final_model.n_jobs = -1          # prediction only; makes permutation importance faster

data = joblib.load(DATA_FILE)
feature_table = pd.read_csv(FEATURES_FILE)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train = np.asarray(data["y_train"], dtype=float)
y_val = np.asarray(data["y_val"], dtype=float)
y_test = np.asarray(data["y_test"], dtype=float)
for name, Xp in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
    assert list(Xp.columns) == list(feature_table["feature"]), f"{name}: columns differ from {FEATURES_FILE}"

saved = pd.read_csv(PRED_FILE)
assert len(saved) == len(X_test), "test_predictions.csv has a different number of rows than X_test"
assert np.array_equal(saved["test_index"].to_numpy(), np.asarray(X_test.index)), \
    "test_predictions.csv rows are not in the same order as X_test"
assert np.allclose(saved["y_true"].to_numpy(), y_test), "y_true in test_predictions.csv differs from y_test"
y_pred = saved["y_pred"].to_numpy(dtype=float)

check = final_model.predict(X_test)
assert np.allclose(check, y_pred, rtol=1e-6, atol=1e-6), \
    "the loaded model does not reproduce test_predictions.csv: wrong model file or files out of sync"
r2_now = r2_score(y_test, y_pred)
assert np.isclose(r2_now, decision["test"]["R2"], rtol=1e-6), "Test R² differs from Part 12"
print(f"OK: loaded {final_name}; reproduces test_predictions.csv and the Part 12 Test R² ({r2_now:.4f}).")
print(f"Part 12 Test: R² {decision['test']['R2']:.4f} | MAE {gbp(decision['test']['MAE'])} | "
      f"RMSE {gbp(decision['test']['RMSE'])} | MedAPE {decision['test']['MedAPE']:.2f}% | "
      f"±5% {decision['test']['within5']:.2f}% | ±10% {decision['test']['within10']:.2f}%")
print("Part 13 describes this model. It does not change it. Any later improvement must use Validation.")
print()

# ---- group the 215 columns back to original features -------------------
feature_groups, cat_labels = {}, {}
for f in X_train.columns:
    g, lab = group_of(f)
    feature_groups.setdefault(g, []).append(f)
    if lab is not None:
        cat_labels[f] = lab
grp_tbl = pd.DataFrame({
    "group": list(feature_groups),
    "n_columns": [len(c) for c in feature_groups.values()],
    "example columns": [", ".join(map(str, c[:2])) for c in feature_groups.values()],
}).sort_values("n_columns", ascending=False)
print(f"{X_train.shape[1]} preprocessed columns -> {len(feature_groups)} original-feature groups:")
print(grp_tbl.to_string(index=False))
print("CHECK THIS TABLE: every one-hot family should be ONE row. If a family is split into many one-column")
print("groups, add its prefix to CATEGORICAL_GROUPS in CONFIG.")
print()

# ---- original values (Make, Model, Year, Mileage, Engine size) ---------
def find_col(df, aliases):
    for c in df.columns:
        if norm(c) in aliases:
            return c
    return None


def looks_scaled(col):
    s = X_train[col]
    return abs(s.mean()) < 0.1 and abs(s.std() - 1) < 0.1


split_src = {}
if SPLIT_FILES:
    for name, yp, Xp in [("Train", y_train, X_train), ("Validation", y_val, X_val), ("Test", y_test, X_test)]:
        f = SPLIT_FILES[name]
        assert os.path.exists(f), f"missing file: {f} (SPLIT_FILES['{name}'])"
        df = pd.read_csv(f)
        assert len(df) == len(Xp), f"{name}: {f} has {len(df):,} rows but X has {len(Xp):,}"
        pcol = find_col(df, ["price"])
        assert pcol is not None, f"{name}: no price column in {f}"
        assert np.allclose(pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float), yp, rtol=1e-6), \
            f"{name}: price in {f} differs from y ({name}) -> the rows are not the same cars in the same order"
        split_src[name] = df.reset_index(drop=True)
    print("OK: original columns for Train / Validation / Test come from their own split files "
          "(row order verified against y).")


def build_context(X, y, name):
    """One row per car with the ORIGINAL features, rebuilt from X (and the split's own file if given)."""
    ctx = pd.DataFrame({"row_id": np.asarray(X.index), "price": np.asarray(y, dtype=float)})
    src = split_src.get(name)      # same length and order as X (checked above), so positional
    # categorical: prefer original text, else rebuild from the one-hot block
    for g, aliases in RAW_CATEGORICAL_ALIASES.items():
        col = find_col(src, aliases) if src is not None else None
        if col is not None:
            ctx[g] = src[col].astype(str).str.strip().to_numpy()
        elif g in feature_groups:
            cols = feature_groups[g]
            block = X[cols].to_numpy()
            labels = np.array([cat_labels[c] for c in cols], dtype=object)
            lab = labels[block.argmax(axis=1)]
            lab[block.max(axis=1) <= 0] = "(no model flag)" if g == "Model" else "(no flag)"
            ctx[g] = lab
    # numeric
    for g, aliases in NUMERIC_ALIASES.items():
        col = find_col(src, aliases) if src is not None else None
        if col is not None:
            ctx[g] = pd.to_numeric(src[col], errors="coerce").to_numpy()
            continue
        col = find_col(X, aliases)
        if col is not None and not looks_scaled(col):
            ctx[g] = X[col].to_numpy(dtype=float)
    if "Age" not in ctx and "Year" in ctx and REFERENCE_YEAR is not None:
        ctx["Age"] = REFERENCE_YEAR - ctx["Year"]
    return ctx


ctx = build_context(X_test, y_test, "Test")
ctx_train = build_context(X_train, y_train, "Train")
ctx["pred"] = y_pred
ctx["err"] = ctx["price"] - ctx["pred"]
ctx["abs_err"] = ctx["err"].abs()
ctx["ape"] = ctx["abs_err"] / ctx["price"] * 100
available = [c for c in ["Make", "Model", "Year", "Age", "Mileage", "Engine size"] if c in ctx]
missing = [c for c in ["Make", "Model", "Year", "Mileage", "Engine size"] if c not in ctx]
print(f"Original values available for the tables: {available}")
for g, aliases in NUMERIC_ALIASES.items():
    c = find_col(X_train, aliases)
    if c is not None and g not in ctx:
        print(f"   {g}: column '{c}' exists but looks standardised (mean≈0, std≈1) -> original units needed (SPLIT_FILES)")
if missing:
    print(f"NOT available: {missing}. The numeric columns look scaled or have unexpected names. "
          "Check SPLIT_FILES in CONFIG (see top of the file); the related tables/panels are skipped until then.")
print()

# ----------------------------------------------------------------------
# 13.1 Actual vs Predicted (Test)
# ----------------------------------------------------------------------
banner("13.1 ACTUAL vs PREDICTED PRICE  (Test, final model)")
train_max, train_min = y_train.max(), y_train.min()
above = ctx[ctx["price"] > train_max]
below = ctx[ctx["price"] < train_min]
print(f"Test rows priced above the Train maximum ({gbp(train_max)}): {len(above)}")
for _, r in above.iterrows():
    print(f"   actual {gbp(r['price'])} -> predicted {gbp(r['pred'])}  (error {gbp(r['err'])})")
print(f"Test rows priced below the Train minimum ({gbp(train_min)}): {len(below)}")
print("A tree model cannot predict outside the price range it was trained on, so these rows are a limit")
print("of the method, not necessarily a mistake of the model. They are reported, not removed.")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, zoom in zip(axes, [False, True]):
    d = ctx[ctx["price"] <= ZOOM_MAX] if zoom else ctx
    hi = ZOOM_MAX if zoom else max(ctx["price"].max(), ctx["pred"].max()) * 1.03
    ax.scatter(d["price"], d["pred"], s=5, alpha=0.25, color="#2a9d8f", linewidths=0)
    ax.plot([0, hi], [0, hi], color="black", lw=1, ls="--", label="perfect prediction (y = x)")
    if not zoom and len(above):
        ax.scatter(above["price"], above["pred"], marker="*", s=160, color="#e63946", zorder=5,
                   label=f"priced above Train max ({len(above)})")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Actual price (£)")
    ax.set_ylabel("Predicted price (£)")
    ax.set_title(f"Zoom: actual ≤ {gbp(ZOOM_MAX)} ({len(d):,} cars)" if zoom
                 else f"All Test cars ({len(ctx):,})")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Part 13: Actual vs Predicted, {final_name}  (Test R² {r2_now:.4f})", fontsize=13)
fig.tight_layout()
fig.savefig("part13_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 13.2 Residuals
# ----------------------------------------------------------------------
banner("13.2 RESIDUAL ANALYSIS  (error = actual - predicted; positive = model predicted too low)")
q = pd.qcut(ctx["pred"], 20, duplicates="drop")
trend = ctx.groupby(q, observed=True).agg(x=("pred", "median"), y=("err", "median"))
print(f"Mean residual {gbp(ctx['err'].mean())} | median residual {gbp(ctx['err'].median())} | "
      f"median |% error| {ctx['ape'].median():.2f}%")
print("Residuals are plotted against PREDICTED price. Plotting them against actual price creates an artificial")
print("funnel (cheap cars look over-predicted, expensive cars under-predicted) because you condition on y.")
print()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
ax.scatter(ctx["pred"], ctx["err"], s=5, alpha=0.25, color="#2a9d8f", linewidths=0)
ax.plot(trend["x"], trend["y"], color="#e63946", lw=2, label="median residual (20 bins)")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Predicted price (£)")
ax.set_ylabel("Actual − Predicted (£)")
ax.set_title("Residual vs predicted price")
ax.legend(fontsize=9)
ax = axes[1]
lo, hi = ctx["err"].quantile([0.01, 0.99])
ax.hist(ctx["err"].clip(lo, hi), bins=60, color="#264653")
ax.axvline(0, color="white", lw=1)
ax.set_xlabel("Actual − Predicted (£), clipped at 1st/99th pct\n(the two edge bars are clipped tails, not real peaks)")
ax.set_title("Distribution of errors")
ax = axes[2]
sgn = (ctx["err"] / ctx["price"] * 100).clip(-30, 30)
ax.hist(sgn, bins=60, color="#264653")
for v in (-10, -5, 5, 10):
    ax.axvline(v, color="#e63946", ls="--", lw=1)
ax.set_xlabel("Signed % error, clipped at ±30% (edge bars = clipped tails)\ndashed = ±5%, ±10%")
ax.set_title("Percentage error")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Part 13: Residual analysis (Test)", fontsize=13)
fig.tight_layout()
fig.savefig("part13_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 13.3 Feature importance (impurity + grouped permutation on Validation)
# ----------------------------------------------------------------------
banner("13.3 FEATURE IMPORTANCE")
imp_ind = None
if hasattr(final_model, "feature_importances_"):
    imp_ind = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp_ind.rename("impurity_importance").to_csv("part13_feature_importance.csv", header=True)
    imp_grp = pd.Series({g: imp_ind[c].sum() for g, c in feature_groups.items()})
else:
    imp_grp = pd.Series({g: np.nan for g in feature_groups})
    print("This model has no feature_importances_; only permutation importance is shown.")

rng = np.random.default_rng(RANDOM_STATE)
n_rows = min(N_PERM_ROWS, len(X_val))
sel = rng.choice(len(X_val), n_rows, replace=False)
Xv, yv = X_val.iloc[sel].reset_index(drop=True), y_val[sel]
base_pred = final_model.predict(Xv)
base_mae, base_r2 = mean_absolute_error(yv, base_pred), r2_score(yv, base_pred)
print(f"Permutation importance on {n_rows:,} Validation rows (baseline MAE {gbp(base_mae)}, R² {base_r2:.4f}).")
print("Each GROUP of columns (all one-hot columns of a feature, plus Age/Year) is shuffled together,")
print(f"{N_PERM_REPEATS} times; importance = how much the error rises when its information is destroyed.")

def perm_table(group_cols):
    rows = []
    for g, cols in group_cols.items():
        d_mae, d_r2 = [], []
        for _ in range(N_PERM_REPEATS):
            idx = rng.permutation(n_rows)
            Xp = Xv.copy()
            Xp[cols] = Xv[cols].to_numpy()[idx]
            p = final_model.predict(Xp)
            d_mae.append(mean_absolute_error(yv, p) - base_mae)
            d_r2.append(base_r2 - r2_score(yv, p))
        rows.append({"group": g, "n_columns": len(cols),
                     "perm_MAE_increase": np.mean(d_mae), "perm_MAE_std": np.std(d_mae),
                     "perm_R2_drop": np.mean(d_r2)})
    return pd.DataFrame(rows)


grouped = perm_table(feature_groups)
grouped.insert(2, "impurity_sum", grouped["group"].map(imp_grp))
grouped = grouped.sort_values("perm_MAE_increase", ascending=False).reset_index(drop=True)
grouped.to_csv("part13_grouped_importance.csv", index=False)

show = grouped.head(12).copy()
print()
print(show.assign(impurity_sum=show["impurity_sum"].map("{:.3f}".format),
                  perm_MAE_increase=show["perm_MAE_increase"].map("£{:,.0f}".format),
                  perm_MAE_std=show["perm_MAE_std"].map("£{:,.0f}".format),
                  perm_R2_drop=show["perm_R2_drop"].map("{:.4f}".format)).to_string(index=False))
print()
fam_cols = {}
for fname, members in FAMILIES.items():
    cols = [c for m in members if m in feature_groups for c in feature_groups[m]]
    if len({m for m in members if m in feature_groups}) >= 2:
        fam_cols[fname] = list(dict.fromkeys(cols))
if fam_cols:
    fam_tbl = perm_table(fam_cols).sort_values("perm_MAE_increase", ascending=False)
    fam_tbl.to_csv("part13_family_importance.csv", index=False)
    print("Families of related features shuffled TOGETHER (information the model gets from that kind of feature):")
    print(fam_tbl.assign(perm_MAE_increase=fam_tbl["perm_MAE_increase"].map("£{:,.0f}".format),
                         perm_MAE_std=fam_tbl["perm_MAE_std"].map("£{:,.0f}".format),
                         perm_R2_drop=fam_tbl["perm_R2_drop"].map("{:.4f}".format)).to_string(index=False))
    print("Compare with the single-feature rows above: a family is worth more than any one member because the")
    print("members partly replace each other (e.g. mileage can be rebuilt from car_age x mileage_per_year).")
    print()
print("How to read it:")
print(" - Impurity importance is computed on the TRAINING data and is biased towards features with many split")
print("   points. Permutation importance is computed on unseen data and measures what the model really uses.")
print(" - 'Important' means the model relies on it, not that it causes the price. Correlated features share")
print("   information, which is why Age and Year are merged into one group.")
print()

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
g12 = grouped.head(12).iloc[::-1]
axes[0].barh(g12["group"], g12["perm_MAE_increase"], xerr=g12["perm_MAE_std"], color="#264653")
axes[0].set_xlabel("Increase in MAE when the group is shuffled (£)")
axes[0].set_title("Grouped permutation importance (Validation)")
if imp_ind is not None:
    gi = imp_grp.sort_values(ascending=False).head(12).iloc[::-1]
    axes[1].barh(gi.index, gi.values, color="#2a9d8f")
    axes[1].set_xlabel("Sum of impurity importance")
    axes[1].set_title("Grouped impurity importance (Train)")
    ti = imp_ind.head(15).iloc[::-1]
    axes[2].barh([strip_prefix(c) for c in ti.index], ti.values, color="#9aa5b1")
    axes[2].set_xlabel("Impurity importance")
    axes[2].set_title("Top 15 individual columns")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Part 13: What does the model rely on?", fontsize=13)
fig.tight_layout()
fig.savefig("part13_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 13.4 Mileage / Age / Make / Engine size  (observed vs what the model predicts)
# ----------------------------------------------------------------------
banner("13.4 MILEAGE, AGE, MANUFACTURER, ENGINE SIZE  (Test, medians)")
print("Median ACTUAL price vs median PREDICTED price per group. These are observed associations;")
print("other features (age, mileage, model, engine ...) differ between groups, so none of this is a cause.")
print()
rel_rows = []


def med_by_bins(col, edges, fmt):
    labels = make_labels(edges, fmt)
    b = pd.cut(ctx[col], bins=edges, labels=labels, right=False)
    out = ctx.groupby(b, observed=False).agg(N=("price", "size"), median_actual=("price", "median"),
                                              median_predicted=("pred", "median"))
    out.index.name = "bin"
    n_out = int(b.isna().sum())
    if n_out:
        print(f"   ({n_out} rows fall outside the {col} bins and are not shown)")
    return out


panels = []
if "Mileage" in ctx:
    panels.append(("Mileage", MILEAGE_EDGES, f_thousand, "Mileage (thousand miles)"))
age_col = "Age" if "Age" in ctx else ("Year" if "Year" in ctx else None)
if age_col == "Age":
    panels.append(("Age", AGE_EDGES, f_plain, "Vehicle age (years)"))
elif age_col == "Year":
    yr = ctx["Year"].dropna()
    panels.append(("Year", [yr.min(), *np.percentile(yr, [20, 40, 60, 80]), np.inf], f_plain, "Year (quantile bins)"))
if "Engine size" in ctx:
    panels.append(("Engine size", ENGINE_EDGES, f_plain, "Engine size (litres)"))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()
ax = axes[0]
if "Make" in ctx:
    mk = ctx.groupby("Make").agg(N=("price", "size"), median_actual=("price", "median"),
                                 median_predicted=("pred", "median"), MAE=("abs_err", "mean"),
                                 MedAPE=("ape", "median"))
    mk = mk[mk["N"] >= MIN_N].sort_values("median_actual", ascending=False)
    print(f"Median price by Make (Test, Make with at least {MIN_N} cars):")
    print(mk.assign(median_actual=mk["median_actual"].map(gbp), median_predicted=mk["median_predicted"].map(gbp),
                    MAE=mk["MAE"].map(gbp), MedAPE=mk["MedAPE"].map("{:.2f}%".format)).to_string())
    print()
    mk_p = mk.iloc[::-1]
    ax.barh(mk_p.index, mk_p["median_actual"], color="#264653", label="median actual")
    ax.scatter(mk_p["median_predicted"], mk_p.index, color="#e9c46a", edgecolor="black", zorder=3,
               label="median predicted")
    ax.set_xlabel("Median price (£)")
    ax.set_title("Manufacturer")
    ax.legend(fontsize=9)
    mk.assign(variable="Make").reset_index().rename(columns={"Make": "bin"}).to_csv(
        "part13_make_summary.csv", index=False)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "Make not available\n(check SPLIT_FILES)", ha="center")
for ax, spec in zip(axes[1:], panels + [None] * 3):
    if spec is None:
        ax.axis("off")
        continue
    col, edges, fmt, xlabel = spec
    t = med_by_bins(col, edges, fmt)
    t = t[t["N"] > 0]
    print(f"Median price by {col}:")
    print(t.assign(median_actual=t["median_actual"].map(gbp), median_predicted=t["median_predicted"].map(gbp)).to_string())
    print()
    ax.plot(t.index.astype(str), t["median_actual"], marker="o", color="#264653", label="median actual")
    ax.plot(t.index.astype(str), t["median_predicted"], marker="s", ls="--", color="#e9c46a",
            label="median predicted")
    for i, n in enumerate(t["N"]):
        ax.annotate(f"n={n}", (i, t["median_actual"].iloc[i]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median price (£)")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    rel_rows.append(t.reset_index().assign(variable=col))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Part 13: Price vs Manufacturer / Mileage / Age / Engine size (Test medians)", fontsize=13)
fig.tight_layout()
fig.savefig("part13_relationships.png", dpi=150, bbox_inches="tight")
plt.show()
if rel_rows:
    pd.concat(rel_rows).to_csv("part13_relationship_bins.csv", index=False)

num_cols = [c for c in ["Mileage", "Age", "Year", "Engine size"] if c in ctx]
if num_cols:
    sp = ctx[["price"] + num_cols].corr(method="spearman")["price"].drop("price")
    print("Spearman rank correlation with actual price (Test; monotonic association, not causation):")
    print(sp.map("{:+.3f}".format).to_string())
    print()

# ----------------------------------------------------------------------
# 13.5 Largest errors
# ----------------------------------------------------------------------
banner("13.5 LARGEST PREDICTION ERRORS  (Test)")
neigh_ok = all(c in ctx and c in ctx_train for c in ["Model"]) and (
    ("Year" in ctx and "Year" in ctx_train) or ("Age" in ctx and "Age" in ctx_train))
if neigh_ok:
    ycol = "Year" if "Year" in ctx and "Year" in ctx_train else "Age"
    counts = ctx_train.assign(_y=ctx_train[ycol].round()).groupby(["Model", "_y"]).size().to_dict()

    def n_neighbours(row):
        y0 = round(row[ycol])
        return sum(counts.get((row["Model"], y0 + d), 0) for d in (-1, 0, 1))

    ctx["train_neighbours"] = ctx.apply(n_neighbours, axis=1)
    print(f"train_neighbours = number of TRAIN cars with the same Model and {ycol} within ±1")
    print("(few neighbours = rare combination = less evidence for the model to learn from)")
    print()

cols = [c for c in ["price", "pred", "err", "abs_err", "ape", "Make", "Model", "Year", "Age", "Mileage",
                    "Engine size", "train_neighbours"] if c in ctx]
fmt_map = {"price": gbp, "pred": gbp, "err": gbp, "abs_err": gbp, "ape": "{:.1f}%".format,
           "Mileage": "{:,.0f}".format, "Engine size": "{:.1f}".format}


def show_errors(df, title, path):
    t = df[["row_id"] + cols]
    t.to_csv(path, index=False)
    print(title)
    disp = df[cols].copy()
    for c, f in fmt_map.items():
        if c in disp:
            disp[c] = disp[c].map(f)
    print(disp.to_string())
    print()


top_abs = ctx.nlargest(TOP_N_ERRORS, "abs_err")
top_pct = ctx.nlargest(TOP_N_ERRORS, "ape")
show_errors(top_abs, f"A. {TOP_N_ERRORS} largest ABSOLUTE errors (cost the most money):", "part13_largest_abs_errors.csv")
show_errors(top_pct, f"B. {TOP_N_ERRORS} largest PERCENTAGE errors (relatively worst):", "part13_largest_pct_errors.csv")

print("How typical are these cars compared with all Test cars? (median values)")
cmp_cols = [c for c in ["price", "Age", "Mileage", "Engine size", "train_neighbours"] if c in ctx]
cmp = pd.DataFrame({"all Test": ctx[cmp_cols].median(),
                    "10 largest abs errors": top_abs[cmp_cols].median(),
                    "10 largest % errors": top_pct[cmp_cols].median()}).T
print(cmp.to_string())
print(f"Share of the 10 largest abs errors that are signed 'predicted too low': "
      f"{(top_abs['err'] > 0).mean() * 100:.0f}% | for % errors: {(top_pct['err'] > 0).mean() * 100:.0f}%")
print("Look for: expensive cars, rare Model/Year combinations (low train_neighbours), unusual mileage or engine,")
print("and cars outside the Train price range. Do not call a row a data error unless the evidence is clear.")
print()

# ----------------------------------------------------------------------
# 13.6 Error by price / age / mileage band
# ----------------------------------------------------------------------
banner("13.6 ERROR BY BAND  (Test)")


def error_table(col, edges, fmt, name):
    labels = make_labels(edges, fmt)
    b = pd.cut(ctx[col], bins=edges, labels=labels, right=False)
    rows = []
    for lab in labels:
        s = ctx[b == lab]
        if len(s) == 0:
            continue
        rows.append({"by": name, "band": lab, "N": len(s), "MAE": s["abs_err"].mean(),
                     "RMSE": float(np.sqrt((s["err"] ** 2).mean())), "MedAPE_pct": s["ape"].median(),
                     "mean_residual": s["err"].mean(), "N_below_min": len(s) < MIN_N})
    return pd.DataFrame(rows)


tables = [error_table("price", PRICE_EDGES, f_money, "actual price"),
          error_table("pred", PRICE_EDGES, f_money, "predicted price")]
if "Age" in ctx:
    tables.append(error_table("Age", AGE_EDGES, f_plain, "age (years)"))
if "Mileage" in ctx:
    tables.append(error_table("Mileage", MILEAGE_EDGES, f_thousand, "mileage (thousand)"))
all_err = pd.concat(tables, ignore_index=True)
all_err.to_csv("part13_error_by_band.csv", index=False)
for name, t in all_err.groupby("by", sort=False):
    print(f"Error by {name}:")
    print(t.drop(columns="by").assign(MAE=t["MAE"].map(gbp), RMSE=t["RMSE"].map(gbp),
                                      MedAPE_pct=t["MedAPE_pct"].map("{:.2f}%".format),
                                      mean_residual=t["mean_residual"].map(gbp)).to_string(index=False))
    print()
print(f"N_below_min = True means fewer than {MIN_N} cars: do not draw conclusions from that row.")
print("Banding by ACTUAL price mixes in selection (you pick rows by their true value); the 'predicted price'")
print("table is the fairer view of where the model is weak. Compare both and mention it in your write-up.")
print()

# ----------------------------------------------------------------------
# 13.7 Log
# ----------------------------------------------------------------------
banner("PART 13 LOG")
print(f"Model: {final_name} (unchanged, nothing retrained). Test used descriptively only.")
if len(grouped):
    top3 = ", ".join(f"{g} (+{gbp(v)} MAE)" for g, v in zip(grouped['group'][:3], grouped['perm_MAE_increase'][:3]))
    print(f"Top 3 groups by permutation importance (Validation): {top3}")
print()
print("Now write YOUR OWN conclusions in the notebook. Suggested checklist:")
print(" 1. Actual vs Predicted: where do points leave the y = x line? What happens above the Train maximum?")
print(" 2. Importance: which groups does the model rely on most? Do impurity and permutation agree?")
print(" 3. Mileage / Age / Make / Engine size: direction and shape of the association (use 'associated with').")
print(" 4. Largest errors: what do they have in common (price level, rare combination, unusual values)?")
print(" 5. Error by band: is the model equally good for cheap, mid-range and expensive cars?")
print(" 6. Limitations: observational data, correlated features, range limit of tree models, small groups.")
print()
print("Saved: part13_actual_vs_predicted.png, part13_residuals.png, part13_feature_importance.png,")
print("       part13_relationships.png, part13_feature_importance.csv, part13_grouped_importance.csv,")
print("       part13_relationship_bins.csv, part13_make_summary.csv, part13_largest_abs_errors.csv,")
print("       part13_largest_pct_errors.csv, part13_error_by_band.csv")