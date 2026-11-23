# =====================================================================
# PART 11: Improve Your Best Model  (Random Forest + RandomizedSearchCV)
# =====================================================================
# Input : preprocessed_data.joblib (Part 7), models/random_forest.joblib (Part 8),
#         baseline_results.csv, part10_cv_results.csv, part10_train_vs_val.csv
# Output: models/random_forest_tuned.joblib
#         part11_before_after.csv, part11_search_results.csv, part11_best_params.csv
#         part11_before_after.png
#
# Rules:
#   1. Tune ONE strong model. Random Forest is chosen from Part 10 (best Val R² / CV R² / Val MAE).
#   2. Search uses the TRAINING set only, via the same 5-fold KFold as Part 10.
#      The Validation set is used AFTER the search, for the Before vs After comparison.
#      The Test set is dropped on load and is not used (Part 12).
#   3. "Before" is the Part 8 Random Forest, not sklearn's library defaults.
#      Part 8 already set n_estimators=200, max_features=0.33, min_samples_leaf=2.
#   4. RandomizedSearchCV (not a full grid). Search is made cheaper in ways that should not
#      change *which* depth / leaf / feature settings win; the final forest is then refit
#      with more trees. See CONFIG.
#   5. Improvement is judged on Validation (+ Deal Advisor bands), not on Training R².
#      A tiny Validation change is reported as such; a worse Validation result is a valid finding.

import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5                         # same splitter protocol as Part 10 (search + final CV)
N_ITER = 12                          # 12 x 5 = 60 cheap fits (space is smaller than before)
SEARCH_N_JOBS = -1                   # keep -1 for speed; set 2 only if Colab workers crash (slower)
REFIT_N_JOBS = -1
SEARCH_N_ESTIMATORS = 150            # proxy during search (ranking max_depth / leaf / max_features)
REFIT_N_ESTIMATORS = 300             # extra trees after the winner is known (RF rarely worse with more)

MODEL_NAME = "Random Forest"
DATA_FILE = "preprocessed_data.joblib"
BASELINE_MODEL_FILE = "models/random_forest.joblib"
TUNED_MODEL_FILE = "models/random_forest_tuned.joblib"

# Search space: the parameters that actually change this forest on THIS dataset.
# n_estimators is NOT searched: it is fixed cheaply during CV, then raised at refit.
#   max_depth        — Part 8 left this at None. Train R² = 0.9826 vs Val R² = 0.9555
#                      (gap 0.027): a depth cap is the direct way to test extra regularisation.
#                      Depth 10 is omitted: it is likely to underfit a model already at Val R² 0.955.
#   min_samples_leaf — Part 8 used 2. 1 can overfit; 4 can shrink the Train–Val gap.
#   min_samples_split— default was 2; 5 / 10 reduce highly specific splits.
#   max_features     — Part 8 chose 0.33 (not 'sqrt') because only 6 of 215 columns are
#                      continuous. 1.0 is excluded: every split sees all columns, trees become
#                      similar, search fits get slower, and diversity (the point of a forest) drops.
#                      0.2 sits between sqrt (~15 cols) and 0.33 (~71 cols).
PARAM_DISTRIBUTIONS = {
    "max_depth": [None, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.2, 0.33],
}
REPORT_PARAM_KEYS = ["n_estimators", *PARAM_DISTRIBUTIONS.keys()]


def slug(model_name):
    return model_name.lower().replace(" ", "_")


def regression_metrics(y_true, y_pred):
    """Same definition as Part 10 (APE relative to the true price)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true * 100
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAPE": float(np.median(ape)),
        "within5": float((ape <= 5).mean() * 100),
        "within10": float((ape <= 10).mean() * 100),
        "neg_preds": int((y_pred < 0).sum()),
    }


def params_of(est):
    return {k: getattr(est, k) for k in REPORT_PARAM_KEYS}


# ----------------------------------------------------------------------
# 11.1 Load Part 7 / Part 8 / Part 10 artifacts  (Test is dropped)
# ----------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

data = joblib.load(DATA_FILE)
data.pop("X_test", None)
data.pop("y_test", None)
X_train, X_val = data["X_train"], data["X_val"]
y_train = np.asarray(data["y_train"], dtype=float)
y_val = np.asarray(data["y_val"], dtype=float)

assert len(X_train) == len(y_train) and len(X_val) == len(y_val)
assert list(X_train.columns) == list(X_val.columns)
assert (y_train > 0).all() and (y_val > 0).all()

baseline_est = joblib.load(BASELINE_MODEL_FILE)
assert type(baseline_est).__name__ == "RandomForestRegressor", type(baseline_est)

baseline_csv = pd.read_csv("baseline_results.csv").set_index("Model")
part10_tv = pd.read_csv("part10_train_vs_val.csv", index_col=0)
part10_cv = pd.read_csv("part10_cv_results.csv", index_col=0)

print(f"Train rows: {len(X_train):,}   |   Validation rows: {len(X_val):,}   |   "
      f"features: {X_train.shape[1]}")
print(f"Model chosen for tuning: {MODEL_NAME}")
print(f"Part 8 baseline params: {params_of(baseline_est)}")
print()

# ----------------------------------------------------------------------
# 11.2 Why this model (facts from Part 10 — not a new ranking)
# ----------------------------------------------------------------------
print("=" * 90)
print("11.2 WHY RANDOM FOREST  (from Part 10; Test set still unused)")
print("=" * 90)
why = pd.DataFrame({
    "Train R²": part10_tv["Train_R2"],
    "Val R²": part10_tv["Val_R2"],
    "R² gap": part10_tv["R2_gap"],
    "Val MAE": part10_tv["Val_MAE"],
    "CV R² mean": part10_cv["CV_R2_mean"],
    "CV R² std": part10_cv["CV_R2_std"],
})
print(why.to_string(float_format=lambda v: f"{v:.4f}"))
print()
print("Selection rule used here: strongest Validation R², CV R², and Validation MAE.")
print("XGBoost is close (Val R² 0.9537 vs 0.9555) but Part 11 asks to improve ONE strong model,")
print("not to switch models. The RF Train–Val gap (0.027) is expected for unpruned trees;")
print("CV R² (0.9560 ± 0.0014) matches Validation, so the model generalises — the gap alone")
print("is not a reason to abandon it.")
print()

# ----------------------------------------------------------------------
# 11.3 BEFORE: Part 8 Random Forest on Train + Validation
# ----------------------------------------------------------------------
# Recompute from the saved model so the table is not typed by hand.
pred_tr_before = np.asarray(baseline_est.predict(X_train), dtype=float)
pred_val_before = np.asarray(baseline_est.predict(X_val), dtype=float)
before_tr = regression_metrics(y_train, pred_tr_before)
before_val = regression_metrics(y_val, pred_val_before)

assert np.allclose(before_val["R2"], baseline_csv.loc[MODEL_NAME, "Val_R2"], rtol=1e-6), \
    "Loaded RF Validation R² differs from baseline_results.csv — files are out of sync"
assert np.allclose(before_val["R2"], part10_tv.loc[MODEL_NAME, "Val_R2"], rtol=1e-6), \
    "Loaded RF Validation R² differs from part10_train_vs_val.csv"

print("=" * 90)
print("11.3 BEFORE TUNING  (Part 8 Random Forest; sklearn defaults are NOT the baseline)")
print("=" * 90)
print(f"Parameters : {params_of(baseline_est)}")
print(f"Train      : R² {before_tr['R2']:.4f} | MAE £{before_tr['MAE']:,.0f} | RMSE £{before_tr['RMSE']:,.0f}")
print(f"Validation : R² {before_val['R2']:.4f} | MAE £{before_val['MAE']:,.0f} | RMSE £{before_val['RMSE']:,.0f}")
print(f"Deal bands : MedAPE {before_val['MedAPE']:.2f}% | ±5% {before_val['within5']:.2f}% | "
      f"±10% {before_val['within10']:.2f}%")
print(f"Part 10 CV : R² {part10_cv.loc[MODEL_NAME, 'CV_R2_mean']:.4f} ± "
      f"{part10_cv.loc[MODEL_NAME, 'CV_R2_std']:.4f}")
print()

# ----------------------------------------------------------------------
# 11.4 RandomizedSearchCV on TRAINING only
# ----------------------------------------------------------------------
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

# n_jobs=1 on the estimator so parallelism is across folds/candidates, not nested.
# n_estimators is fixed during search: extra trees after the winner is known, not while ranking.
search_estimator = RandomForestRegressor(
    n_estimators=SEARCH_N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

print("=" * 90)
print("11.4 RANDOMIZED SEARCH  (Training folds only; Validation is held out)")
print("=" * 90)
print(f"n_iter={N_ITER}  |  cv={CV_FOLDS}  |  scoring=R² (refit)  |  "
      f"max fits = {N_ITER * CV_FOLDS}")
print(f"Search n_estimators = {SEARCH_N_ESTIMATORS} (proxy). "
      f"Winner is refit with n_estimators = {REFIT_N_ESTIMATORS}.")
print("Search CV R² is therefore NOT comparable to Part 10 (that forest used 200 trees).")
print("The After 5-fold CV in 11.6 is the fair comparison.")
print("Parameter distributions:")
for k, v in PARAM_DISTRIBUTIONS.items():
    print(f"  {k}: {v}")
print()

t0 = time.time()
search = RandomizedSearchCV(
    estimator=search_estimator,
    param_distributions=PARAM_DISTRIBUTIONS,
    n_iter=N_ITER,
    cv=cv,
    scoring=scoring,
    refit="r2",
    random_state=RANDOM_STATE,
    n_jobs=SEARCH_N_JOBS,
    verbose=1,
    return_train_score=False,
)
search.fit(X_train, y_train)
search_seconds = time.time() - t0

cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_r2")
keep_cols = [c for c in cv_results.columns if c.startswith("param_")
             or c in ("mean_test_r2", "std_test_r2", "mean_test_mae", "std_test_mae",
                      "mean_test_rmse", "std_test_rmse", "rank_test_r2",
                      "mean_fit_time")]
cv_results[keep_cols].to_csv("part11_search_results.csv", index=False)

best_params = dict(search.best_params_)
print()
print(f"Search finished in {search_seconds / 60:.1f} min")
print(f"Best params by mean CV R² : {best_params}")
print(f"Best CV R² (search)       : {search.best_score_:.4f}")
print()

# ----------------------------------------------------------------------
# 11.5 Refit the winner on the FULL training set  (n_jobs=-1)
# ----------------------------------------------------------------------
# Search used a cheaper n_estimators. Raise it here; extra trees almost never hurt a forest.
refit_params = {**best_params, "n_estimators": REFIT_N_ESTIMATORS}
tuned = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=REFIT_N_JOBS, **refit_params)
t0 = time.time()
tuned.fit(X_train, y_train)
print(f"Refit tuned RF on full Train ({len(X_train):,} rows) in {time.time() - t0:.1f}s")
joblib.dump(tuned, TUNED_MODEL_FILE, compress=3)
print(f"Saved {TUNED_MODEL_FILE} ({os.path.getsize(TUNED_MODEL_FILE) / 1e6:.1f} MB)")
print()

# ----------------------------------------------------------------------
# 11.6 AFTER: Train + Validation metrics + 5-fold CV (same protocol as Part 10)
# ----------------------------------------------------------------------
pred_tr_after = np.asarray(tuned.predict(X_train), dtype=float)
pred_val_after = np.asarray(tuned.predict(X_val), dtype=float)
assert np.isfinite(pred_tr_after).all() and np.isfinite(pred_val_after).all()
after_tr = regression_metrics(y_train, pred_tr_after)
after_val = regression_metrics(y_val, pred_val_after)

print("Re-running 5-fold CV on the tuned hyperparameters (Training only, unfitted clones) ...")
t0 = time.time()
cv_after = cross_validate(
    clone(tuned), X_train, y_train, cv=cv,
    scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
    n_jobs=SEARCH_N_JOBS,
)
after_cv = {
    "CV_R2_mean": cv_after["test_r2"].mean(),
    "CV_R2_std": cv_after["test_r2"].std(),
    "CV_MAE_mean": -cv_after["test_mae"].mean(),
    "CV_MAE_std": cv_after["test_mae"].std(),
    "CV_RMSE_mean": -cv_after["test_rmse"].mean(),
    "CV_RMSE_std": cv_after["test_rmse"].std(),
}
print(f"  CV done ({time.time() - t0:.0f}s)  R² = {after_cv['CV_R2_mean']:.4f} ± {after_cv['CV_R2_std']:.4f}")
print()

# ----------------------------------------------------------------------
# 11.7 Before vs After table
# ----------------------------------------------------------------------
before_cv_r2 = float(part10_cv.loc[MODEL_NAME, "CV_R2_mean"])
before_cv_r2_std = float(part10_cv.loc[MODEL_NAME, "CV_R2_std"])
before_cv_mae = float(part10_cv.loc[MODEL_NAME, "CV_MAE_mean"])
before_cv_rmse = float(part10_cv.loc[MODEL_NAME, "CV_RMSE_mean"])

rows = []
for split, b, a in [
    ("Train", before_tr, after_tr),
    ("Validation", before_val, after_val),
]:
    for metric in ["R2", "MAE", "RMSE", "MedAPE", "within5", "within10"]:
        rows.append({
            "Split": split, "Metric": metric,
            "Before": b[metric], "After": a[metric],
            "Change": a[metric] - b[metric],
        })
rows.append({"Split": "CV (train folds)", "Metric": "R2",
             "Before": before_cv_r2, "After": after_cv["CV_R2_mean"],
             "Change": after_cv["CV_R2_mean"] - before_cv_r2})
rows.append({"Split": "CV (train folds)", "Metric": "MAE",
             "Before": before_cv_mae, "After": after_cv["CV_MAE_mean"],
             "Change": after_cv["CV_MAE_mean"] - before_cv_mae})
rows.append({"Split": "CV (train folds)", "Metric": "RMSE",
             "Before": before_cv_rmse, "After": after_cv["CV_RMSE_mean"],
             "Change": after_cv["CV_RMSE_mean"] - before_cv_rmse})

cmp = pd.DataFrame(rows)
cmp.to_csv("part11_before_after.csv", index=False)

pd.DataFrame([
    {"stage": "before", **params_of(baseline_est)},
    {"stage": "after", **params_of(tuned)},
]).to_csv("part11_best_params.csv", index=False)


def fmt_cell(metric, v):
    if metric == "R2":
        return f"{v:.4f}"
    if metric in ("MedAPE", "within5", "within10"):
        return f"{v:.2f}%"
    return f"£{v:,.0f}"


shown = pd.DataFrame({
    "Metric": [
        "Validation R²", "Validation MAE", "Validation RMSE",
        "Validation MedAPE", "Validation ±5%", "Validation ±10%",
        "Train R²", "CV R² (mean)", "CV MAE (mean)",
    ],
    "Before": [
        fmt_cell("R2", before_val["R2"]), fmt_cell("MAE", before_val["MAE"]),
        fmt_cell("RMSE", before_val["RMSE"]), fmt_cell("MedAPE", before_val["MedAPE"]),
        fmt_cell("within5", before_val["within5"]), fmt_cell("within10", before_val["within10"]),
        fmt_cell("R2", before_tr["R2"]), f"{before_cv_r2:.4f} ± {before_cv_r2_std:.4f}",
        fmt_cell("MAE", before_cv_mae),
    ],
    "After": [
        fmt_cell("R2", after_val["R2"]), fmt_cell("MAE", after_val["MAE"]),
        fmt_cell("RMSE", after_val["RMSE"]), fmt_cell("MedAPE", after_val["MedAPE"]),
        fmt_cell("within5", after_val["within5"]), fmt_cell("within10", after_val["within10"]),
        fmt_cell("R2", after_tr["R2"]), f"{after_cv['CV_R2_mean']:.4f} ± {after_cv['CV_R2_std']:.4f}",
        fmt_cell("MAE", after_cv["CV_MAE_mean"]),
    ],
    "Change": [
        f"{after_val['R2'] - before_val['R2']:+.4f}",
        f"£{after_val['MAE'] - before_val['MAE']:+,.0f}",
        f"£{after_val['RMSE'] - before_val['RMSE']:+,.0f}",
        f"{after_val['MedAPE'] - before_val['MedAPE']:+.2f} pp",
        f"{after_val['within5'] - before_val['within5']:+.2f} pp",
        f"{after_val['within10'] - before_val['within10']:+.2f} pp",
        f"{after_tr['R2'] - before_tr['R2']:+.4f}",
        f"{after_cv['CV_R2_mean'] - before_cv_r2:+.4f}",
        f"£{after_cv['CV_MAE_mean'] - before_cv_mae:+,.0f}",
    ],
})
print("=" * 90)
print("11.7 BEFORE vs AFTER  (Validation is the comparison that matters; Test was not used)")
print("=" * 90)
print(shown.to_string(index=False))
print()
print("Parameters")
print(f"  Before : {params_of(baseline_est)}")
print(f"  After  : {params_of(tuned)}")
print()

# ----------------------------------------------------------------------
# 11.8 Did tuning actually improve the model?
# ----------------------------------------------------------------------
d_r2 = after_val["R2"] - before_val["R2"]
d_mae = after_val["MAE"] - before_val["MAE"]
d_rmse = after_val["RMSE"] - before_val["RMSE"]
d_w5 = after_val["within5"] - before_val["within5"]
d_gap = (after_tr["R2"] - after_val["R2"]) - (before_tr["R2"] - before_val["R2"])

# Practical thresholds: RF CV std in Part 10 was 0.0014. Changes smaller than ~1 std
# on R², or a few pounds on MAE, are noise rather than a "major improvement".
R2_NOISE = 0.002
MAE_NOISE = 15.0  # pounds

val_better = (d_r2 > R2_NOISE) or (d_mae < -MAE_NOISE)
val_worse = (d_r2 < -R2_NOISE) or (d_mae > MAE_NOISE)
overfit_worse = (after_tr["R2"] - after_val["R2"]) > (before_tr["R2"] - before_val["R2"] + 0.005)

print("=" * 90)
print("11.8 INTERPRETATION")
print("=" * 90)
if val_better and not overfit_worse:
    verdict = "YES — modest/real improvement on Validation"
    explain = (f"Validation R² changed by {d_r2:+.4f} and MAE by £{d_mae:+,.0f}. "
               "That is large enough to treat as an improvement on held-out data, "
               "not only a higher Training score.")
elif val_worse:
    verdict = "NO — tuning did not beat the Part 8 baseline on Validation"
    explain = (f"Validation R² changed by {d_r2:+.4f} and MAE by £{d_mae:+,.0f}. "
               "The Part 8 configuration already used informed (not library-default) "
               "hyperparameters. Keep it as the candidate for Part 12 unless you re-run "
               "the search with a different space.")
else:
    verdict = "NEGLIGIBLE — within the noise of 5-fold CV"
    explain = (f"Validation R² changed by {d_r2:+.4f} and MAE by £{d_mae:+,.0f}. "
               f"Part 10's CV R² std for this model was {before_cv_r2_std:.4f}. "
               "A change this small should not be described as a major improvement.")

print(f"Verdict: {verdict}")
print(explain)
print(f"Train–Val R² gap change (After gap − Before gap): {d_gap:+.4f} "
      f"({'larger gap' if d_gap > 0 else 'smaller or equal gap'})")
print(f"Deal Advisor ±5% band: {before_val['within5']:.2f}% → {after_val['within5']:.2f}% "
      f"({d_w5:+.2f} pp). A small R² change can still matter here if ±5% moves.")
print()
print("Part 12 should pick between the Part 8 RF and this tuned RF using Validation "
      "(and overfitting), then evaluate the winner ONCE on Test.")
print()

# ----------------------------------------------------------------------
# 11.9 Figure: Before vs After on Validation (R² / MAE / RMSE)
# ----------------------------------------------------------------------
C_BEFORE, C_AFTER = "#9aa5b1", "#2a9d8f"
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
panels = [
    ("R²  (higher is better)", [before_val["R2"], after_val["R2"]], "{:.4f}", True),
    ("MAE in £  (lower is better)", [before_val["MAE"], after_val["MAE"]], "£{:,.0f}", False),
    ("RMSE in £  (lower is better)", [before_val["RMSE"], after_val["RMSE"]], "£{:,.0f}", False),
]
for ax, (title, vals, fmt, is_r2) in zip(axes, panels):
    bars = ax.bar(["Before\n(Part 8 RF)", "After\n(tuned RF)"], vals,
                  color=[C_BEFORE, C_AFTER], width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), " " + fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if is_r2:
        lo = min(0.90, min(vals) - 0.01)
        ax.set_ylim(lo, 1.0)
    else:
        ax.set_ylim(0, max(vals) * 1.18)
fig.suptitle("Part 11: Random Forest before vs after hyperparameter tuning (Validation)", fontsize=13)
fig.tight_layout()
fig.savefig("part11_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 11.10 Notebook paragraph (paste into Colab; numbers are from THIS run)
# ----------------------------------------------------------------------
print("=" * 90)
print("PART 11 LOG / NOTEBOOK PARAGRAPH")
print("=" * 90)
print(
    f"Random Forest was selected for hyperparameter tuning because it achieved the strongest "
    f"validation and cross-validation performance in Part 10 (Val R² {before_val['R2']:.4f}, "
    f"mean 5-fold CV R² {before_cv_r2:.4f} ± {before_cv_r2_std:.4f}, Val MAE "
    f"£{before_val['MAE']:,.0f}). RandomizedSearchCV explored {N_ITER} random combinations of "
    f"max_depth, min_samples_split, min_samples_leaf, and max_features on the training set only, "
    f"using the same {CV_FOLDS}-fold shuffled split as Part 10 (random_state={RANDOM_STATE}). "
    f"n_estimators was fixed at {SEARCH_N_ESTIMATORS} during the search (a cheaper ranking proxy) "
    f"and raised to {REFIT_N_ESTIMATORS} when the winner was refit on the full training set. "
    f"max_features=1.0 was excluded so trees keep split-level diversity. "
    f"The validation set was not used to pick hyperparameters. "
    f"The search selected {best_params} and the refit used {refit_params}. "
    f"On the held-out validation set the tuned model reached R² {after_val['R2']:.4f}, "
    f"MAE £{after_val['MAE']:,.0f}, RMSE £{after_val['RMSE']:,.0f} "
    f"(before: R² {before_val['R2']:.4f}, MAE £{before_val['MAE']:,.0f}, "
    f"RMSE £{before_val['RMSE']:,.0f}). {explain} "
    f"Verdict: {verdict}. The test set was not used."
)
print()
print("Saved: models/random_forest_tuned.joblib, part11_before_after.csv, "
      "part11_search_results.csv, part11_best_params.csv, part11_before_after.png")
