# =====================================================================
# PART 9: Evaluate & Compare Models  (Validation set only)
# =====================================================================
# Input : baseline_results.csv, validation_predictions.csv      (output of Part 8)
# Output: part9_comparison_table.csv, part9_model_comparison.png
#
# Rules:
#   1. Validation only. The Test set is not loaded here (it is used once, in Part 12).
#   2. R2 / MAE / RMSE are recomputed from the saved predictions and checked against Part 8,
#      so the table is real evaluation output, not numbers typed by hand.
#   3. No model is selected here. This is a baseline comparison:
#      overfitting -> Part 10, tuning -> Part 11, final choice + Test -> Part 12.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_ORDER = ["Linear Regression", "Decision Tree", "Random Forest",
               "Gradient Boosting", "XGBoost"]          # same order as the assignment table


def slug(model_name):
    return model_name.lower().replace(" ", "_")          # same rule as Part 8


# ----------------------------------------------------------------------
# 9.1 Load Part 8 outputs and recompute the three required metrics
# ----------------------------------------------------------------------
baseline = pd.read_csv("baseline_results.csv").set_index("Model").loc[MODEL_ORDER]
preds = pd.read_csv("validation_predictions.csv")
y_true = preds["y_true"].to_numpy(dtype=float)
print(f"Validation rows: {len(preds):,}   |   models: {len(MODEL_ORDER)}")

rows = {}
for m in MODEL_ORDER:
    y_pred = preds[f"pred_{slug(m)}"].to_numpy(dtype=float)
    assert np.isfinite(y_pred).all(), f"{m}: non-finite predictions"
    rows[m] = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
table = pd.DataFrame(rows).T
table.index.name = "Model"

# the recomputed numbers must match what Part 8 saved
for col, ref in [("R2", "Val_R2"), ("MAE", "Val_MAE"), ("RMSE", "Val_RMSE")]:
    assert np.allclose(table[col], baseline[ref], rtol=1e-6), \
        f"{col}: recomputed values differ from baseline_results.csv -- files are out of sync"
print("OK: metrics recomputed from validation_predictions.csv match baseline_results.csv")
print()

# ----------------------------------------------------------------------
# 9.2 Required comparison table  (Validation)
# ----------------------------------------------------------------------
best = {"R2": table["R2"].idxmax(), "MAE": table["MAE"].idxmin(), "RMSE": table["RMSE"].idxmin()}

shown = pd.DataFrame({
    "R2": table["R2"].map("{:.4f}".format),
    "MAE": table["MAE"].map("£{:,.1f}".format),
    "RMSE": table["RMSE"].map("£{:,.1f}".format),
})
for col, m in best.items():
    shown.loc[m, col] += " ★"                      # ★ = best value in that column
shown = shown.rename(columns={"R2": "R²"})

print("=" * 70)
print("9.2 MODEL COMPARISON  (Validation set; ★ = best in column)")
print("=" * 70)
print(shown.to_string())
print()

# rank per metric (1 = best): the ordering is not identical across metrics
ranks = pd.DataFrame({
    "R2": table["R2"].rank(ascending=False),
    "MAE": table["MAE"].rank(ascending=True),
    "RMSE": table["RMSE"].rank(ascending=True),
}).astype(int)
print("Rank per metric (1 = best):")
print(ranks.rename(columns={"R2": "R²"}).to_string())
print()

# ----------------------------------------------------------------------
# 9.3 One figure, three panels: R2, MAE, RMSE
# ----------------------------------------------------------------------
panels = [
    ("R2", "R²  (higher is better)\naxis starts at 0.80", "{:.4f}", (0.80, 1.0)),
    ("MAE", "MAE in £  (lower is better)", "£{:,.0f}", None),
    ("RMSE", "RMSE in £  (lower is better)", "£{:,.0f}", None),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, (col, title, fmt, xlim) in zip(axes, panels):
    vals = table[col]
    best_model = best[col]
    colors = ["#2a9d8f" if m == best_model else "#9aa5b1" for m in vals.index]
    bars = ax.barh(vals.index, vals.values, color=colors)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt.format(v),
                va="center", fontsize=9)
    ax.set_xlim(xlim if xlim else (0, vals.max() * 1.2))
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()                             # Linear Regression on top, like the table
fig.suptitle("Baseline models on the Validation set (green = best)", fontsize=13)
fig.tight_layout()
fig.savefig("part9_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 9.4 Supplementary metrics for the Deal Advisor (not part of the required table)
# ----------------------------------------------------------------------
supp = baseline[["Val_MedAPE_pct", "Val_within5_pct", "Val_within10_pct", "n_negative_val_predictions"]]
supp = supp.rename(columns={"Val_MedAPE_pct": "MedAPE %", "Val_within5_pct": "within ±5 %",
                            "Val_within10_pct": "within ±10 %",
                            "n_negative_val_predictions": "negative preds"})
print("=" * 70)
print("9.4 SUPPLEMENTARY (relevant to the Deal Advisor's ±5% / ±10% bands) -- Validation")
print("=" * 70)
print(supp.to_string(float_format=lambda v: f"{v:,.2f}"))
print()

# ----------------------------------------------------------------------
# 9.5 Save + log
# ----------------------------------------------------------------------
table.to_csv("part9_comparison_table.csv")

print("=" * 70)
print("PART 9 LOG")
print("=" * 70)
print(f"Best R²  : {best['R2']} ({table.loc[best['R2'], 'R2']:.4f})")
print(f"Best MAE : {best['MAE']} (£{table.loc[best['MAE'], 'MAE']:,.1f})")
print(f"Best RMSE: {best['RMSE']} (£{table.loc[best['RMSE'], 'RMSE']:,.1f})")
print("This is a baseline comparison on Validation only. The Test set was not used, and no final "
      "model is chosen yet (Part 10: overfitting, Part 11: tuning, Part 12: final selection).")
print("Saved: part9_comparison_table.csv, part9_model_comparison.png")
