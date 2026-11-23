# =====================================================================
# PART 10: Check for Overfitting  (Training vs Validation)
# =====================================================================
# Input : preprocessed_data.joblib (Part 7) + models/<model>.joblib (Part 8),
#         baseline_results.csv, validation_predictions.csv (Part 8)
# Output: part10_train_vs_val.csv, part10_cv_results.csv, part10_overfitting_check.png
#         (+ part10_learning_curves.png when RUN_LEARNING_CURVE = True)
#
# Rules:
#   1. Diagnosis only. No hyperparameter is changed here (tuning -> Part 11),
#      and no final model is chosen here (-> Part 12).
#   2. Validation + Training only. The Test set is not loaded (it is used once, in Part 12).
#   3. Validation numbers are recomputed and checked against Part 8 / Part 9 (same assert style).
#   4. No artificial "good / bad" threshold on the gap. The conclusion for each model is built from
#      several pieces of evidence: Train vs Val gap + 5-fold CV + the Validation level itself.
#   5. The 5-fold CV uses the Part 7 output X_train (already preprocessed) and an unfitted clone of each
#      Part 8 model. Part 7 fitted its preprocessing once on the whole Training set, so it is NOT re-fitted
#      inside each fold (see the NOTE printed in 10.3).

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# CONFIG  (edit only this block if your Part 7 / Part 8 file names are different)
# ----------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5                 # same splitter will be reused in Part 11
RUN_CV = True                # 5-fold CV on Training only (takes a few minutes on Colab)
CV_N_JOBS = -1
RUN_LEARNING_CURVE = False   # optional, heavier: only for the models listed below
LEARNING_CURVE_MODELS = ["Random Forest", "XGBoost"]

DATA_FILE = "preprocessed_data.joblib"   # Part 7 output (the same file Part 8 loaded)
MODEL_FILE = "models/{slug}.joblib"      # Part 8 output: one fitted model per file

MODEL_ORDER = ["Linear Regression", "Decision Tree", "Random Forest",
               "Gradient Boosting", "XGBoost"]          # same order as Part 9


def slug(model_name):
    return model_name.lower().replace(" ", "_")          # same rule as Part 8 / Part 9


def regression_metrics(y_true, y_pred):
    """R2 / MAE / RMSE (required) + the Deal Advisor metrics (APE is relative to the true price)."""
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


# ----------------------------------------------------------------------
# 10.1 Load Part 8 artifacts
# ----------------------------------------------------------------------
baseline = pd.read_csv("baseline_results.csv").set_index("Model").loc[MODEL_ORDER]
preds = pd.read_csv("validation_predictions.csv")
y_val = preds["y_true"].to_numpy(dtype=float)

data = joblib.load(DATA_FILE)
data.pop("X_test", None)    # the Test set is dropped on load: it must not be touched before Part 12
data.pop("y_test", None)
X_train, X_val = data["X_train"], data["X_val"]
y_train = np.asarray(data["y_train"], dtype=float)
y_val_part7 = np.asarray(data["y_val"], dtype=float)

assert len(X_train) == len(y_train), "X_train / y_train length mismatch"
assert len(X_val) == len(y_val), "X_val does not match validation_predictions.csv"
assert np.allclose(y_val_part7, y_val), "y_val in preprocessed_data.joblib differs from validation_predictions.csv"
assert list(X_train.columns) == list(X_val.columns), "X_train and X_val have different columns"
assert (y_train > 0).all() and (y_val > 0).all(), "non-positive prices found (APE would be undefined)"

# Part 8 saved the bare fitted models (not Pipelines): they expect the already-preprocessed X from Part 7
pipes = {m: joblib.load(MODEL_FILE.format(slug=slug(m))) for m in MODEL_ORDER}
print(f"Train rows: {len(X_train):,}   |   Validation rows: {len(X_val):,}   |   "
      f"features: {X_train.shape[1]}   |   models: {len(pipes)}")
print()

# ----------------------------------------------------------------------
# 10.2 Training vs Validation metrics
# ----------------------------------------------------------------------
train_m, val_m = {}, {}
for m, pipe in pipes.items():
    p_tr = np.asarray(pipe.predict(X_train), dtype=float)
    p_val = np.asarray(pipe.predict(X_val), dtype=float)
    assert np.isfinite(p_tr).all() and np.isfinite(p_val).all(), f"{m}: non-finite predictions"
    # the loaded model must be the same one that produced validation_predictions.csv in Part 8
    saved = preds[f"pred_{slug(m)}"].to_numpy(dtype=float)
    assert np.allclose(p_val, saved, rtol=1e-4, atol=1e-2), \
        f"{m}: loaded model gives different validation predictions than validation_predictions.csv"
    train_m[m] = regression_metrics(y_train, p_tr)
    val_m[m] = regression_metrics(y_val, p_val)
train_m, val_m = pd.DataFrame(train_m).T, pd.DataFrame(val_m).T

# Validation numbers must match Part 8 / Part 9
for col, ref in [("R2", "Val_R2"), ("MAE", "Val_MAE"), ("RMSE", "Val_RMSE")]:
    assert np.allclose(val_m[col], baseline[ref], rtol=1e-6), \
        f"{col}: Validation values differ from baseline_results.csv -- files are out of sync"
print("OK: validation metrics match baseline_results.csv (and Part 9)")
soft = [("MedAPE", "Val_MedAPE_pct"), ("within5", "Val_within5_pct"), ("within10", "Val_within10_pct")]
if all(np.allclose(val_m[c], baseline[r], rtol=1e-3, atol=0.05) for c, r in soft):
    print("OK: MedAPE / within +-5% / within +-10% use the same definition as Part 8")
else:
    print("NOTE: MedAPE / within-band values differ slightly from Part 8 (different definition?). "
          "Train and Validation here use the same formula, so the comparison inside Part 10 is still fair.")
print()

comp = pd.DataFrame(index=pd.Index(MODEL_ORDER, name="Model"))
for k in ["R2", "MAE", "RMSE"]:
    comp[f"Train_{k}"] = train_m[k]
    comp[f"Val_{k}"] = val_m[k]
comp["R2_gap"] = comp["Train_R2"] - comp["Val_R2"]          # Train - Val
comp["MAE_ratio"] = comp["Val_MAE"] / comp["Train_MAE"].replace(0, np.nan)      # Val / Train
comp["RMSE_ratio"] = comp["Val_RMSE"] / comp["Train_RMSE"].replace(0, np.nan)   # NaN if Train error is exactly 0


def ratio_fmt(v):
    return "n/a (train=0)" if pd.isna(v) else f"{v:.2f}x"

for k in ["MedAPE", "within5", "within10", "neg_preds"]:
    comp[f"Train_{k}"] = train_m[k]
    comp[f"Val_{k}"] = val_m[k]

shown = pd.DataFrame({
    "Train R²": comp["Train_R2"].map("{:.4f}".format),
    "Val R²": comp["Val_R2"].map("{:.4f}".format),
    "R² gap": comp["R2_gap"].map("{:+.4f}".format),
    "Train MAE": comp["Train_MAE"].map("£{:,.0f}".format),
    "Val MAE": comp["Val_MAE"].map("£{:,.0f}".format),
    "MAE ratio": comp["MAE_ratio"].map(ratio_fmt),
    "Train RMSE": comp["Train_RMSE"].map("£{:,.0f}".format),
    "Val RMSE": comp["Val_RMSE"].map("£{:,.0f}".format),
    "RMSE ratio": comp["RMSE_ratio"].map(ratio_fmt),
})
print("=" * 90)
print("10.2 TRAINING vs VALIDATION   (R² gap = Train - Val;  ratio = Val / Train)")
print("=" * 90)
print(shown.to_string())
print()

adv = pd.DataFrame({
    "Train MedAPE %": comp["Train_MedAPE"], "Val MedAPE %": comp["Val_MedAPE"],
    "Train ±5 %": comp["Train_within5"], "Val ±5 %": comp["Val_within5"],
    "Train ±10 %": comp["Train_within10"], "Val ±10 %": comp["Val_within10"],
    "Train neg preds": comp["Train_neg_preds"], "Val neg preds": comp["Val_neg_preds"],
})
print("=" * 90)
print("10.2b DEAL ADVISOR METRICS  (share of cars predicted within the ±5% / ±10% bands)")
print("=" * 90)
adv_fmt = {c: (lambda v: f"{v:,.0f}") if "neg preds" in c else (lambda v: f"{v:,.2f}") for c in adv.columns}
print(adv.to_string(formatters=adv_fmt))
print()

# ----------------------------------------------------------------------
# 10.3 5-fold cross-validation on the TRAINING set only
# ----------------------------------------------------------------------
cv_tbl = None
if RUN_CV:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows = {}
    print("NOTE: CV runs on the Part 7 output. The preprocessing (scaler / encoder) was fitted once on the whole "
          "Training set, so it is not re-fitted inside each fold. The effect is small for one-hot + scaling; "
          "say so in the report. No Validation or Test row is involved.")
    for m in MODEL_ORDER:
        t0 = time.time()
        res = cross_validate(clone(pipes[m]), X_train, y_train, cv=cv, scoring=scoring, n_jobs=CV_N_JOBS)
        rows[m] = {
            "CV_R2_mean": res["test_r2"].mean(), "CV_R2_std": res["test_r2"].std(),
            "CV_MAE_mean": -res["test_mae"].mean(), "CV_MAE_std": res["test_mae"].std(),
            "CV_RMSE_mean": -res["test_rmse"].mean(), "CV_RMSE_std": res["test_rmse"].std(),
        }
        print(f"  CV done: {m:<18} ({time.time() - t0:5.0f}s)   R² = {rows[m]['CV_R2_mean']:.4f}")
    cv_tbl = pd.DataFrame(rows).T
    cv_tbl.index.name = "Model"

    shown_cv = pd.DataFrame({
        "CV R² (mean ± std)": [f"{r.CV_R2_mean:.4f} ± {r.CV_R2_std:.4f}" for r in cv_tbl.itertuples()],
        "CV MAE (mean ± std)": [f"£{r.CV_MAE_mean:,.0f} ± {r.CV_MAE_std:,.0f}" for r in cv_tbl.itertuples()],
        "CV RMSE (mean ± std)": [f"£{r.CV_RMSE_mean:,.0f} ± {r.CV_RMSE_std:,.0f}" for r in cv_tbl.itertuples()],
    }, index=cv_tbl.index)
    print()
    print("=" * 90)
    print(f"10.3 {CV_FOLDS}-FOLD CROSS-VALIDATION  (Training set only; fresh unfitted clone of each model per fold)")
    print("=" * 90)
    print(shown_cv.to_string())
    print()
else:
    print("10.3 skipped (RUN_CV = False)")
    print()

# ----------------------------------------------------------------------
# 10.4 Split sanity checklist (short)
# ----------------------------------------------------------------------
def row_hashes(X, y):
    d = X.copy()
    d["__price__"] = y
    return pd.util.hash_pandas_object(d, index=False).to_numpy()


h_tr, h_val = row_hashes(X_train, y_train), row_hashes(X_val, y_val)
n_overlap = int(np.isin(h_val, h_tr).sum())
n_dup_train = int(pd.Series(h_tr).duplicated().sum())
price_cols = [c for c in X_train.columns if "price" in str(c).lower()]

print("=" * 90)
print("10.4 SPLIT / LEAKAGE SANITY CHECKLIST")
print("=" * 90)
print(f"[auto]   Feature columns containing 'price'          : {price_cols if price_cols else 'none  OK'}")
print(f"[auto]   Identical rows (features + Price) Train∩Val : {n_overlap}   {'OK' if n_overlap == 0 else 'CHECK: duplicates were not removed before the split'}")
print(f"[auto]   Duplicate rows inside Train                 : {n_dup_train}   {'OK' if n_dup_train == 0 else 'CHECK: duplicates remain after Part 3'}")
print(f"[auto]   Train : Validation = {len(X_train):,} : {len(X_val):,}  (ratio {len(X_train) / len(X_val):.2f}, expected ≈ 70:15 = 4.67)")
print("[manual] Preprocessing was fitted on Training only (Part 7)        -> confirm in your notebook")
print("[manual] Validation / Test never used for fitting or cleaning rules -> confirm in your notebook")
print("[note]   Test rows are not loaded here; the Train∩Test check belongs to Part 12, right before the one-time test evaluation.")
print()

# ----------------------------------------------------------------------
# 10.5 One figure, three panels: R2, MAE, RMSE  (Train vs [CV] vs Validation)
# ----------------------------------------------------------------------
C_TRAIN, C_CV, C_VAL = "#9aa5b1", "#e9c46a", "#2a9d8f"
series = [("Training", "Train", C_TRAIN, None)]
if cv_tbl is not None:
    series.append((f"{CV_FOLDS}-fold CV (train only, mean ± std)", "CV", C_CV, "cv"))
series.append(("Validation", "Val", C_VAL, None))

all_r2 = list(comp["Train_R2"]) + list(comp["Val_R2"])
if cv_tbl is not None:
    all_r2 += list(cv_tbl["CV_R2_mean"] - cv_tbl["CV_R2_std"])
r2_lo = max(np.floor((min(all_r2) - 0.02) * 20) / 20, 0.0)

panels = [("R2", f"R²  (higher is better)\naxis starts at {r2_lo:.2f}", "{:.3f}"),
          ("MAE", "MAE in £  (lower is better)", "£{:,.0f}"),
          ("RMSE", "RMSE in £  (lower is better)", "£{:,.0f}")]

n = len(MODEL_ORDER)
ypos = np.arange(n)
bar_h = 0.8 / len(series)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
for ax, (k, title, fmt) in zip(axes, panels):
    xmax = 0.0
    for i, (label, prefix, color, kind) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * bar_h
        if kind == "cv":
            vals, err = cv_tbl[f"CV_{k}_mean"].to_numpy(), cv_tbl[f"CV_{k}_std"].to_numpy()
        else:
            vals, err = comp[f"{prefix}_{k}"].to_numpy(), np.zeros(n)
        ax.barh(ypos + off, vals, height=bar_h * 0.92, color=color, label=label,
                xerr=err if kind == "cv" else None, error_kw=dict(ecolor="black", lw=1, capsize=2))
        for yi, v, e in zip(ypos + off, vals, err):
            ax.text(v + e, yi, " " + fmt.format(v), va="center", fontsize=7.5)
        xmax = max(xmax, float((vals + err).max()))
    if k == "R2":
        ax.set_xlim(r2_lo, 1.0 + (1.0 - r2_lo) * 0.22)
        ax.set_xticks(np.arange(r2_lo, 1.0001, 0.05))
    else:
        ax.set_xlim(0, xmax * 1.25)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(MODEL_ORDER)
axes[0].invert_yaxis()                               # Linear Regression on top, like Part 9
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=10)
fig.suptitle("Overfitting check: Training vs Validation (no model selected here)", fontsize=13)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig("part10_overfitting_check.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 10.6 OPTIONAL: learning curves (only the models in LEARNING_CURVE_MODELS)
# ----------------------------------------------------------------------
if RUN_LEARNING_CURVE:
    lc_models = [m for m in LEARNING_CURVE_MODELS if m in pipes]
    fig2, axes2 = plt.subplots(1, len(lc_models), figsize=(6.2 * len(lc_models), 4.4), squeeze=False)
    for ax, m in zip(axes2[0], lc_models):
        t0 = time.time()
        sizes, tr_s, cv_s = learning_curve(
            clone(pipes[m]), X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5),
            cv=KFold(3, shuffle=True, random_state=RANDOM_STATE), scoring="r2", n_jobs=CV_N_JOBS)
        for scores, color, label in [(tr_s, C_TRAIN, "Training score"), (cv_s, C_VAL, "Cross-validation score")]:
            ax.plot(sizes, scores.mean(axis=1), "o-", color=color, label=label)
            ax.fill_between(sizes, scores.mean(axis=1) - scores.std(axis=1),
                            scores.mean(axis=1) + scores.std(axis=1), color=color, alpha=0.2)
        ax.set_title(f"{m}: learning curve (R²)")
        ax.set_xlabel("Number of training rows")
        ax.set_ylabel("R²")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False)
        print(f"  learning curve done: {m} ({time.time() - t0:.0f}s)")
    fig2.tight_layout()
    fig2.savefig("part10_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

# ----------------------------------------------------------------------
# 10.7 Evidence summary (facts only -- the written conclusion per model is yours)
# ----------------------------------------------------------------------
ev = pd.DataFrame({"Train R²": comp["Train_R2"], "Val R²": comp["Val_R2"], "R² gap": comp["R2_gap"]})
if cv_tbl is not None:
    ev["CV R² mean"] = cv_tbl["CV_R2_mean"]
    ev["CV R² std"] = cv_tbl["CV_R2_std"]
    ev["Val - CV"] = comp["Val_R2"] - cv_tbl["CV_R2_mean"]
ev["Val/Train MAE"] = comp["MAE_ratio"]
print("=" * 90)
print("10.7 EVIDENCE SUMMARY  (read all columns together, not the gap alone)")
print("=" * 90)
print(ev.to_string(float_format=lambda v: f"{v:.4f}"))
print()

comp.to_csv("part10_train_vs_val.csv")
if cv_tbl is not None:
    cv_tbl.to_csv("part10_cv_results.csv")

print("=" * 90)
print("PART 10 LOG")
print("=" * 90)
for m in MODEL_ORDER:
    line = (f"{m:<18} Train R² {comp.loc[m, 'Train_R2']:.4f} | Val R² {comp.loc[m, 'Val_R2']:.4f} "
            f"| gap {comp.loc[m, 'R2_gap']:+.4f} | Val MAE £{comp.loc[m, 'Val_MAE']:,.0f} "
            f"({ratio_fmt(comp.loc[m, 'MAE_ratio'])} Train)")
    if cv_tbl is not None:
        line += f" | CV R² {cv_tbl.loc[m, 'CV_R2_mean']:.4f} ± {cv_tbl.loc[m, 'CV_R2_std']:.4f}"
    print(line)
print()
print("How to read this (write your own conclusion per model in the notebook):")
print(" - A large Train-Val gap suggests overfitting only if Validation / CV performance is also weaker "
      "than it could be. A big gap with strong Validation + CV is not by itself a failure.")
print(" - Decision Tree / Random Forest: Training scores are optimistic by construction (the trees have seen "
      "those rows), so judge them mainly by Validation and CV.")
print(" - Linear Regression: a small gap with a lower Validation level points to underfitting, not to a "
      "good model. A small gap alone means little.")
print(" - No numeric threshold is applied. If you use one in your report, state it as your own practical rule.")
print()
print("Training + Validation only. The Test set was not used, no hyperparameters were changed and no final "
      "model is chosen (Part 11: tuning, Part 12: final selection).")
print("Saved: part10_train_vs_val.csv" + (", part10_cv_results.csv" if cv_tbl is not None else "")
      + ", part10_overfitting_check.png" + (", part10_learning_curves.png" if RUN_LEARNING_CURVE else ""))
