# =====================================================================
# PART 12: Select the Final Model + ONE-TIME Test Evaluation
# =====================================================================
# Input : baseline_results.csv (Part 8), part10_cv_results.csv (Part 10),
#         part11_before_after.csv (Part 11), models/*.joblib,
#         preprocessed_data.joblib + feature_names.csv (Part 7)
# Output: part12_model_selection.csv, part12_final_model.json, part12_test_results.csv,
#         test_predictions.csv, part12_final_model.png
#
# Rules:
#   1. The final model is chosen from Validation / CV / Train evidence ONLY. Section 12.1 reads
#      CSV files; the Test set is not loaded until the decision has been written to disk.
#   2. The selection rule is fixed in CONFIG, before any Test number exists. It reuses the
#      noise thresholds from Part 11 (section 11.8). Do not edit them after seeing Test results.
#   3. Only the selected model is evaluated on Test, once. The other candidates never see Test.
#   4. Test results are REPORTED, not tuned against. If Test is worse than Validation the model
#      is not swapped: the difference is a finding to explain, not a reason to re-select.

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# CONFIG  (fixed BEFORE the Test set is touched)
# ----------------------------------------------------------------------
RANDOM_STATE = 42
R2_NOISE = 0.002             # same practical noise thresholds as Part 11 (section 11.8):
MAE_NOISE = 15.0             #   a gain smaller than this is treated as "not material"
N_BOOT = 1000                # bootstrap resamples for the Test confidence intervals
FINAL_MODEL_OVERRIDE = None  # e.g. "Random Forest (tuned)". Set ONLY before the first Test run,
                             # and write the reason in the report. None = use the rule below.
ALLOW_TEST_RERUN = False     # True only if the run crashed AFTER Test was evaluated (e.g. in the plot)

DATA_FILE = "preprocessed_data.joblib"
FEATURES_FILE = "feature_names.csv"
DECISION_FILE = "part12_final_model.json"
TEST_RESULTS_FILE = "part12_test_results.csv"

BASELINE_MODELS = ["Linear Regression", "Decision Tree", "Random Forest",
                   "Gradient Boosting", "XGBoost"]
TUNED_NAME = "Random Forest (tuned)"
CANDIDATES = BASELINE_MODELS + [TUNED_NAME]


def slug(model_name):
    return model_name.lower().replace(" ", "_")


MODEL_FILES = {m: f"models/{slug(m)}.joblib" for m in BASELINE_MODELS}
MODEL_FILES[TUNED_NAME] = "models/random_forest_tuned.joblib"


def regression_metrics(y_true, y_pred):
    """Same definitions as Parts 8 / 10 / 11 (APE relative to the true price)."""
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


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOT, seed=RANDOM_STATE):
    """95% percentile CI from resampling the evaluation rows. Covers test-sample noise only
    (not the randomness of re-training the model)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    err = y_pred - y_true
    ape = np.abs(err) / y_true * 100
    draws = {k: np.empty(n_boot) for k in ["R2", "MAE", "RMSE", "MedAPE", "within5", "within10"]}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        yt, e, a = y_true[i], err[i], ape[i]
        draws["R2"][b] = 1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)
        draws["MAE"][b] = np.mean(np.abs(e))
        draws["RMSE"][b] = np.sqrt(np.mean(e ** 2))
        draws["MedAPE"][b] = np.median(a)
        draws["within5"][b] = np.mean(a <= 5) * 100
        draws["within10"][b] = np.mean(a <= 10) * 100
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def fmt(metric, v):
    if metric == "R2":
        return f"{v:.4f}"
    if metric in ("MedAPE", "within5", "within10"):
        return f"{v:.2f}%"
    return f"£{v:,.0f}"


# ----------------------------------------------------------------------
# 12.1 Build the candidate table from Parts 8 / 10 / 11  (no Test, no data loading)
# ----------------------------------------------------------------------
base = pd.read_csv("baseline_results.csv").set_index("Model").loc[BASELINE_MODELS]
cv10 = pd.read_csv("part10_cv_results.csv", index_col=0).loc[BASELINE_MODELS]
p11 = pd.read_csv("part11_before_after.csv")


def p11_get(split, metric, col="After"):
    row = p11[(p11["Split"] == split) & (p11["Metric"] == metric)]
    assert len(row) == 1, f"part11_before_after.csv: missing row {split}/{metric}"
    return float(row[col].iloc[0])


# the Part 11 'Before' must be the Part 8 Random Forest, otherwise the files are out of sync
assert np.isclose(p11_get("Validation", "R2", "Before"), base.loc["Random Forest", "Val_R2"], rtol=1e-6), \
    "part11_before_after.csv 'Before' differs from baseline_results.csv -- files are out of sync"

rows = {}
for m in BASELINE_MODELS:
    rows[m] = {
        "Train_R2": base.loc[m, "Train_R2"], "Train_MAE": base.loc[m, "Train_MAE"],
        "Train_RMSE": base.loc[m, "Train_RMSE"],
        "Val_R2": base.loc[m, "Val_R2"], "Val_MAE": base.loc[m, "Val_MAE"],
        "Val_RMSE": base.loc[m, "Val_RMSE"], "Val_MedAPE": base.loc[m, "Val_MedAPE_pct"],
        "Val_within5": base.loc[m, "Val_within5_pct"], "Val_within10": base.loc[m, "Val_within10_pct"],
        "CV_R2": cv10.loc[m, "CV_R2_mean"],
    }
rows[TUNED_NAME] = {
    "Train_R2": p11_get("Train", "R2"), "Train_MAE": p11_get("Train", "MAE"),
    "Train_RMSE": p11_get("Train", "RMSE"),
    "Val_R2": p11_get("Validation", "R2"), "Val_MAE": p11_get("Validation", "MAE"),
    "Val_RMSE": p11_get("Validation", "RMSE"), "Val_MedAPE": p11_get("Validation", "MedAPE"),
    "Val_within5": p11_get("Validation", "within5"), "Val_within10": p11_get("Validation", "within10"),
    "CV_R2": p11_get("CV (train folds)", "R2"),
}
cand = pd.DataFrame(rows).T.astype(float)
cand.index.name = "Model"
cand["R2_gap"] = cand["Train_R2"] - cand["Val_R2"]
cand["Val_minus_CV_R2"] = cand["Val_R2"] - cand["CV_R2"]
cand["File_MB"] = [os.path.getsize(MODEL_FILES[m]) / 1e6 if os.path.exists(MODEL_FILES[m]) else np.nan
                   for m in CANDIDATES]

print("=" * 90)
print("12.1 FINAL MODEL SELECTION  (Validation / CV / Train evidence only; Test not loaded yet)")
print("=" * 90)
print(pd.DataFrame({
    "Val R²": cand["Val_R2"].map("{:.4f}".format),
    "Val MAE": cand["Val_MAE"].map("£{:,.0f}".format),
    "Val RMSE": cand["Val_RMSE"].map("£{:,.0f}".format),
    "Train R²": cand["Train_R2"].map("{:.4f}".format),
    "R² gap": cand["R2_gap"].map("{:+.4f}".format),
    "CV R²": cand["CV_R2"].map("{:.4f}".format),
    "Val−CV": cand["Val_minus_CV_R2"].map("{:+.4f}".format),
    "File MB": cand["File_MB"].map(lambda v: "n/a" if pd.isna(v) else f"{v:,.1f}"),
}).to_string())
print()

# --- the rule (fixed above in CONFIG) ---------------------------------
ranks = pd.DataFrame({
    "Val_R2": cand["Val_R2"].rank(ascending=False),
    "Val_MAE": cand["Val_MAE"].rank(ascending=True),
    "Val_RMSE": cand["Val_RMSE"].rank(ascending=True),
})
avg_rank = ranks.mean(axis=1)
leader = cand.assign(avg_rank=avg_rank).sort_values(["avg_rank", "Val_R2"], ascending=[True, False]).index[0]

d_r2 = cand.loc[leader, "Val_R2"] - cand["Val_R2"]          # how far each candidate is below the leader
d_mae = cand["Val_MAE"] - cand.loc[leader, "Val_MAE"]       # how much higher its MAE is
materially_worse = (d_r2 > R2_NOISE) | (d_mae > MAE_NOISE)
tie_set = list(cand.index[~materially_worse])

tie_tbl = cand.loc[tie_set].assign(size=cand.loc[tie_set, "File_MB"].fillna(np.inf))
tie_tbl = tie_tbl.sort_values(["R2_gap", "size"])
rule_pick = tie_tbl.index[0]

print("Selection rule (fixed before Test):")
print("  1. Leader = best average rank on Validation R², MAE and RMSE.")
print(f"  2. A candidate is 'materially worse' than the leader if its Validation R² is lower by more than "
      f"{R2_NOISE} or its MAE is higher by more than £{MAE_NOISE:.0f} (the Part 11 noise thresholds).")
print("  3. Candidates that are NOT materially worse form the tie set. Within the tie set the final model is the")
print("     one that generalises best (smallest Train–Val R² gap), then the smaller model file.")
print()
print(pd.DataFrame({
    "avg rank": avg_rank.map("{:.2f}".format),
    "R² vs leader": (-d_r2).map("{:+.4f}".format),
    "MAE vs leader": d_mae.map("£{:+,.0f}".format),
    "materially worse?": materially_worse.map({True: "yes", False: "no"}),
}).to_string())
print()
print(f"Leader on Validation accuracy : {leader}")
print(f"Tie set (not materially worse): {tie_set}")
print(f"Rule pick (gap, then file size): {rule_pick}")

final_name = rule_pick
if FINAL_MODEL_OVERRIDE is not None:
    assert FINAL_MODEL_OVERRIDE in CANDIDATES, f"unknown candidate: {FINAL_MODEL_OVERRIDE}"
    final_name = FINAL_MODEL_OVERRIDE
    print(f"OVERRIDE active -> final model = {final_name}. Explain this choice in the report.")
if len(tie_set) > 1:
    others = [m for m in tie_set if m != final_name]
    print(f"NOTE: {final_name} is not materially different from {others} on Validation. "
          "The choice rests on the tie-break (generalisation gap, file size), not on accuracy.")
print(f"\nFINAL MODEL: {final_name}")
print()

selection = cand.join(ranks.add_prefix("rank_")).assign(
    avg_rank=avg_rank, R2_vs_leader=-d_r2, MAE_vs_leader=d_mae,
    materially_worse=materially_worse, in_tie_set=cand.index.isin(tie_set),
    selected=(cand.index == final_name))
selection.to_csv("part12_model_selection.csv")

# ----------------------------------------------------------------------
# Decision is now fixed. Guard against a second Test evaluation, then write it to disk.
# ----------------------------------------------------------------------
prev = json.load(open(DECISION_FILE)) if os.path.exists(DECISION_FILE) else None
if os.path.exists(TEST_RESULTS_FILE):
    if not ALLOW_TEST_RERUN:
        raise SystemExit(f"STOP: {TEST_RESULTS_FILE} already exists -- the Test set was already evaluated once. "
                         "Do not evaluate it again. (If the run crashed after Test, set ALLOW_TEST_RERUN = True.)")
    if prev is not None and prev.get("final_model") != final_name:
        raise SystemExit(f"STOP: the saved decision was '{prev.get('final_model')}' but this run selected "
                         f"'{final_name}'. The final model cannot change after Test was evaluated.")

decision = {
    "final_model": final_name, "model_file": MODEL_FILES[final_name],
    "selected_by": "override" if FINAL_MODEL_OVERRIDE is not None else "rule",
    "rule": {"R2_NOISE": R2_NOISE, "MAE_NOISE_gbp": MAE_NOISE,
             "leader": leader, "tie_set": tie_set},
    "validation": {k: float(cand.loc[final_name, f"Val_{k}"]) for k in ["R2", "MAE", "RMSE"]},
    "train_r2": float(cand.loc[final_name, "Train_R2"]),
    "cv_r2_mean": float(cand.loc[final_name, "CV_R2"]),
    "test": None,
}
json.dump(decision, open(DECISION_FILE, "w"), indent=2)
print(f"Decision saved to {DECISION_FILE} (before the Test set is loaded).")
print()

# ----------------------------------------------------------------------
# 12.2 Load the Test set and run the split / leakage checks (promised in Part 10)
# ----------------------------------------------------------------------
data = joblib.load(DATA_FILE)
feature_table = pd.read_csv(FEATURES_FILE)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train = np.asarray(data["y_train"], dtype=float)
y_val = np.asarray(data["y_val"], dtype=float)
y_test = np.asarray(data["y_test"], dtype=float)

for name, Xp, yp in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    assert isinstance(Xp, pd.DataFrame) and len(Xp) == len(yp), f"{name}: X/y problem"
    assert list(Xp.columns) == list(feature_table["feature"]), f"{name}: columns differ from {FEATURES_FILE}"
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in Xp.dtypes), f"{name}: non-numeric columns"
    assert not Xp.isnull().any().any() and not np.isnan(yp).any(), f"{name}: NaN found"
    assert (yp > 0).all(), f"{name}: non-positive prices"


def row_hashes(X, y):
    d = X.copy()
    d["__price__"] = y
    return pd.util.hash_pandas_object(d, index=False).to_numpy()


h_tr, h_val, h_te = row_hashes(X_train, y_train), row_hashes(X_val, y_val), row_hashes(X_test, y_test)
n_tr_te = int(np.isin(h_te, h_tr).sum())
n_val_te = int(np.isin(h_te, h_val).sum())
n_dup_te = int(pd.Series(h_te).duplicated().sum())
n_above = int((y_test > y_train.max()).sum())

print("=" * 90)
print("12.2 TEST SET LOADED  -- split / leakage checks")
print("=" * 90)
print(f"Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,}   "
      f"(Train:Test ratio {len(X_train) / len(X_test):.2f}, expected ≈ 4.67)")
print(f"Test price £{y_test.min():,.0f} - £{y_test.max():,.0f} (mean £{y_test.mean():,.0f})")
print(f"Identical rows (features + Price) Train∩Test : {n_tr_te}   {'OK' if n_tr_te == 0 else 'LEAK'}")
print(f"Identical rows (features + Price) Val∩Test   : {n_val_te}   {'OK' if n_val_te == 0 else 'LEAK'}")
print(f"Duplicate rows inside Test                   : {n_dup_te}   {'OK' if n_dup_te == 0 else 'CHECK'}")
print(f"Test rows priced above the Train maximum (£{y_train.max():,.0f}): {n_above}"
      f"   (tree models cannot predict above the range they were trained on)")
assert n_tr_te == 0 and n_val_te == 0, "Test rows also appear in Train/Validation: fix the split before evaluating"
print()

# ----------------------------------------------------------------------
# 12.3 Load the final model and prove it is the one that was selected
# ----------------------------------------------------------------------
final_model = joblib.load(MODEL_FILES[final_name])
pv = np.asarray(final_model.predict(X_val), dtype=float)
vm = regression_metrics(y_val, pv)
for k in ["R2", "MAE", "RMSE"]:
    assert np.isclose(vm[k], cand.loc[final_name, f"Val_{k}"], rtol=1e-6), \
        f"{final_name}: loaded model gives a different Validation {k} than the selection table"
print(f"OK: the loaded {final_name} reproduces its Validation metrics "
      f"(R² {vm['R2']:.4f}, MAE £{vm['MAE']:,.0f}, RMSE £{vm['RMSE']:,.0f}).")
print()

# ----------------------------------------------------------------------
# 12.4 THE one-time Test evaluation (selected model only)
# ----------------------------------------------------------------------
pt = np.asarray(final_model.predict(X_test), dtype=float)
assert np.isfinite(pt).all(), "non-finite Test predictions"
tm = regression_metrics(y_test, pt)
ci = bootstrap_ci(y_test, pt)

pd.DataFrame([{"Metric": k, "Validation": vm[k], "Test": tm[k],
               "Test_CI_low": ci[k][0] if k in ci else np.nan,
               "Test_CI_high": ci[k][1] if k in ci else np.nan}
              for k in ["R2", "MAE", "RMSE", "MedAPE", "within5", "within10", "neg_preds"]]
             ).assign(model=final_name).to_csv(TEST_RESULTS_FILE, index=False)   # also the one-time marker
pd.DataFrame({"test_index": X_test.index, "y_true": y_test, "y_pred": pt}).to_csv("test_predictions.csv", index=False)

print("=" * 90)
print(f"12.4 FINAL TEST RESULTS  --  {final_name}  (evaluated once)")
print("=" * 90)
print(f"{'Model':28s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s}")
print(f"{final_name + ' (Test)':28s} {tm['R2']:8.4f} {'£' + format(tm['MAE'], ',.0f'):>10s} "
      f"{'£' + format(tm['RMSE'], ',.0f'):>10s}")
print()
print(f"{'Metric':14s} {'Validation':>11s} {'Test':>11s}   Test 95% CI")
for k, label in [("R2", "R²"), ("MAE", "MAE"), ("RMSE", "RMSE"),
                 ("MedAPE", "MedAPE"), ("within5", "within ±5%"), ("within10", "within ±10%")]:
    inside = ci[k][0] <= vm[k] <= ci[k][1]
    print(f"{label:14s} {fmt(k, vm[k]):>11s} {fmt(k, tm[k]):>11s}   "
          f"[{fmt(k, ci[k][0])}, {fmt(k, ci[k][1])}]   Validation value inside CI: {'yes' if inside else 'no'}")
print(f"Negative predictions on Test: {tm['neg_preds']}")
print(f"Train R² {cand.loc[final_name, 'Train_R2']:.4f} | Validation R² {vm['R2']:.4f} | Test R² {tm['R2']:.4f}")
print()

decision["test"] = {k: float(tm[k]) for k in tm}
decision["test_ci95"] = {k: list(v) for k, v in ci.items()}
json.dump(decision, open(DECISION_FILE, "w"), indent=2)

# ----------------------------------------------------------------------
# 12.5 Figure: Train / Validation / Test for the final model
# ----------------------------------------------------------------------
C_TRAIN, C_VAL, C_TEST = "#9aa5b1", "#2a9d8f", "#264653"
panels = [("R2", "R²  (higher is better)"), ("MAE", "MAE in £  (lower is better)"),
          ("RMSE", "RMSE in £  (lower is better)")]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
for ax, (k, title) in zip(axes, panels):
    vals = [cand.loc[final_name, f"Train_{k}"], vm[k], tm[k]]
    err = np.array([[0, 0, tm[k] - ci[k][0]], [0, 0, ci[k][1] - tm[k]]])
    bars = ax.bar(["Train", "Validation", "Test"], vals, color=[C_TRAIN, C_VAL, C_TEST], width=0.62,
                  yerr=err, error_kw=dict(ecolor="black", lw=1, capsize=3))
    for bar, v, up in zip(bars, vals, err[1]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + up, " " + fmt(k, v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if k == "R2":
        ax.set_ylim(max(0.0, np.floor((min(vals) - 0.02) * 20) / 20), 1.0)
        ax.set_title(title + "\n(axis does not start at 0)", fontsize=11)
    else:
        ax.set_ylim(0, max(vals) * 1.2)
fig.suptitle(f"Part 12: {final_name} on Train / Validation / Test  (error bar = Test 95% bootstrap CI)",
             fontsize=13)
fig.tight_layout()
fig.savefig("part12_final_model.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 12.6 Log
# ----------------------------------------------------------------------
size_mb = os.path.getsize(MODEL_FILES[final_name]) / 1e6
print("=" * 90)
print("PART 12 LOG")
print("=" * 90)
print(f"Final model: {final_name} ({size_mb:,.1f} MB, {MODEL_FILES[final_name]}); "
      f"selected by {'override' if FINAL_MODEL_OVERRIDE is not None else 'the pre-declared rule'}.")
print(f"Validation: R² {vm['R2']:.4f}, MAE £{vm['MAE']:,.0f}, RMSE £{vm['RMSE']:,.0f}   |   "
      f"Test: R² {tm['R2']:.4f}, MAE £{tm['MAE']:,.0f}, RMSE £{tm['RMSE']:,.0f}")
print(f"Deal Advisor view on Test: MedAPE {tm['MedAPE']:.2f}% | within ±5% {tm['within5']:.2f}% | "
      f"within ±10% {tm['within10']:.2f}%")
if size_mb > 100:
    print(f"DEPLOYMENT NOTE: the model file is {size_mb:,.0f} MB. GitHub rejects single files over 100 MB, so "
          "plan for Git LFS or external hosting for the Streamlit app (Part 15).")
print()
print("How to read this (write your own conclusion in the notebook):")
print(" - R² is the share of price variance explained. It is NOT an accuracy percentage.")
print(" - The Test number is reported as it is. Do not change the model, features or rule after seeing it.")
print(" - The intervals show Test-sample uncertainty only, not the randomness of re-training the model.")
print(" - Looking at Test errors in Part 13 is descriptive. Any change to the model after that would")
print("   mean the Test set is no longer untouched; use Validation if you want to iterate.")
print()
print(f"Saved: part12_model_selection.csv, {DECISION_FILE}, {TEST_RESULTS_FILE}, "
      "test_predictions.csv, part12_final_model.png")
