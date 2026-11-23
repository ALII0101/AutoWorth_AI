# =====================================================================
# PART 8: Train Baseline Regression Models
# =====================================================================
# Input : preprocessed_data.joblib, feature_names.csv        (output of Part 7)
# Output: models/<model>.joblib          -> كل موديل متدرّب في ملف لوحده
#         baseline_results.csv           -> R2 / MAE / RMSE (Train + Validation) + الأوقات
#         validation_predictions.csv     -> y_true + تنبؤات كل موديل على Validation
#
# القواعد:
#   1. كل الموديلات بتتدرّب على X_train/y_train بس. Validation للتقييم بس.
#   2. الـ Test set بيتشال من الـ dict أول ما الملف يتحمّل -- مش بيتلمس في الجزء ده خالص
#      (هيتقيّم مرة واحدة في Part 12 على الموديل النهائي بس).
#   3. نفس الـ X ونفس الـ preprocessing لكل الموديلات: الفرق الوحيد بين الصفوف هو الموديل.
#   4. الـ price مش بيتحوّل (no log). لو جرّبنا log-target بعدين تبقى تجربة منفصلة بتتحكم عليها
#      Validation، والتنبؤات بتترجع بـ expm1 قبل حساب الأخطاء بالجنيه.
#   5. الإعدادات هنا baseline معقولة ومش نتيجة tuning. الـ tuning في Part 11 لأفضل موديل بس.

import os
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as err:
    raise ImportError("xgboost is required for Part 8 -- run: pip install xgboost") from err

SEED = 42
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 8.0 Load + validate the Part 7 outputs
# ----------------------------------------------------------------------
data = joblib.load("preprocessed_data.joblib")
data.pop("X_test")      # الـ Test اتشال عمداً: مستحيل نستخدمه بالغلط في الجزء ده
data.pop("y_test")
X_train, X_val = data["X_train"], data["X_val"]
y_train, y_val = data["y_train"], data["y_val"]
feature_table = pd.read_csv("feature_names.csv")

print("=" * 70)
print("8.0 LOAD + VALIDATE  (Test set intentionally NOT loaded into this script's variables)")
print("=" * 70)
for name, Xp, yp in [("Train", X_train, y_train), ("Validation", X_val, y_val)]:
    assert isinstance(Xp, pd.DataFrame), f"{name}: expected a pandas DataFrame, got {type(Xp)}"
    assert len(Xp) == len(yp), f"{name}: X/y length mismatch"
    assert Xp.shape[1] == len(feature_table), f"{name}: column count != feature_names.csv"
    assert list(Xp.columns) == list(feature_table["feature"]), f"{name}: columns differ from feature_names.csv"
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in Xp.dtypes), f"{name}: non-numeric columns found"
    assert not Xp.isnull().any().any(), f"{name}: NaN in X"
    assert not yp.isnull().any(), f"{name}: NaN in y"
    assert (yp > 0).all(), f"{name}: non-positive prices found -- percentage errors would be undefined"
    mem_mb = Xp.memory_usage(deep=True).sum() / 1e6
    print(f"  {name:10s}: X {Xp.shape} | dense DataFrame | {mem_mb:7.1f} MB | "
          f"price £{yp.min():,.0f} - £{yp.max():,.0f} (mean £{yp.mean():,.0f})")
assert list(X_train.columns) == list(X_val.columns)

# XGBoost مابيقبلش أسامي أعمدة فيها [ ] <  -- نتأكد دلوقتي بدل ما نكتشف بعد ما الموديلات التانية تخلص
bad_names = [c for c in X_train.columns if any(ch in c for ch in "[]<")]
assert not bad_names, f"Feature names XGBoost rejects (contain [ ] or <): {bad_names[:5]}"
print("OK: dense numeric DataFrames, no NaN, price > 0, columns identical in both splits, "
      "feature names are XGBoost-safe.")
print()

# ----------------------------------------------------------------------
# 8.1 Define the 5 baseline models
# ----------------------------------------------------------------------
# Linear Regression : مفيش tuning. ملحوظة: الـ one-hot في Part 7 من غير drop، يعني الأعمدة
#                     مرتبطة خطياً (dummy trap). sklearn بيحلّها بـ least-squares (min-norm) فالتنبؤات
#                     سليمة، بس الـ coefficients مش قابلة للتفسير المباشر.
# Decision Tree     : max_depth/min_samples_leaf عشان الشجرة ماتحفظش الـ Train.
# Random Forest     : max_features=0.33 مش 'sqrt'. عندنا 215 عمود منهم 208 dummy و 6 continuous بس؛
#                     sqrt = 14 عمود في كل split، واحتمال إن الـ split مايشوفش ولا continuous
#                     (car_age/mileage/engineSize...) حوالي 66%. 0.33 (= p/3، توصية Breiman للـ regression)
#                     بيخلّي الاحتمال ~9%. ده استنتاج حسابي مش قياس -- Part 11 هيقارن القيم على Validation.
# Gradient Boosting : lr=0.1 × 300 trees × depth 5. إعداد lr=0.05 × 200 × depth 3 كان هيطلع
#                     underfit غالباً (مجهود boosting أقل بكتير من XGBoost) فالمقارنة ماتبقاش عادلة.
# XGBoost           : n_estimators ثابت بدون early stopping على Validation، عشان Validation
#                     مايدخلش في التدريب ويفضل مقياس نضيف للمقارنة.
MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=20, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_features=0.33, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=5, min_samples_leaf=5,
        loss="squared_error", random_state=SEED),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror", n_estimators=500, learning_rate=0.05,
        max_depth=6, min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0, reg_lambda=1, random_state=SEED, n_jobs=-1),
}


def slug(model_name):
    return model_name.lower().replace(" ", "_")


def regression_metrics(y_true, y_pred):
    """R2 / MAE / RMSE هي المطلوبة. MedAPE و within-x% إضافات مرتبطة بـ Smart Deal Advisor
    (التصنيفات بتاعته بالنسبة المئوية: ±5% / ±10%)، ومضمونة تتحسب لأن price > 0 متأكد منه فوق."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),   # sqrt يدوي: شغال على كل نسخ sklearn
        "MedAPE_pct": 100 * float(np.median(ape)),
        "within5_pct": 100 * float(np.mean(ape <= 0.05)),
        "within10_pct": 100 * float(np.mean(ape <= 0.10)),
    }


# ----------------------------------------------------------------------
# 8.2 Train (Train only) -> predict Train + Validation -> metrics -> save
# ----------------------------------------------------------------------
print("=" * 70)
print("8.2 TRAIN ON TRAIN ONLY, PREDICT TRAIN + VALIDATION")
print("=" * 70)
rows = []
val_predictions = pd.DataFrame({"val_index": y_val.index, "y_true": y_val.to_numpy()})

for model_name, model in MODELS.items():
    print(f"  Training {model_name} ...", flush=True)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)              # <- الـ fit الوحيد لكل موديل. Validation مش داخل.
    fit_seconds = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    t0 = time.perf_counter()
    pred_val = model.predict(X_val)
    val_predict_seconds = time.perf_counter() - t0

    assert np.isfinite(pred_train).all() and np.isfinite(pred_val).all(), \
        f"{model_name}: non-finite predictions"

    m_train = regression_metrics(y_train, pred_train)
    m_val = regression_metrics(y_val, pred_val)
    rows.append({
        "Model": model_name,
        **{f"Train_{k}": v for k, v in m_train.items()},
        **{f"Val_{k}": v for k, v in m_val.items()},
        "R2_gap_train_minus_val": m_train["R2"] - m_val["R2"],
        "n_negative_val_predictions": int((pred_val < 0).sum()),
        "fit_seconds": fit_seconds,
        "val_predict_seconds": val_predict_seconds,
    })
    val_predictions[f"pred_{slug(model_name)}"] = pred_val

    path = os.path.join(MODELS_DIR, f"{slug(model_name)}.joblib")
    joblib.dump(model, path, compress=3)
    size_mb = os.path.getsize(path) / 1e6
    print(f"    done in {fit_seconds:7.1f}s | Val R2 = {m_val['R2']:.4f} | "
          f"saved {path} ({size_mb:.1f} MB)", flush=True)
print()

results = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 8.3 Results (Validation = المقارنة الرسمية، Train = للـ overfitting check)
# ----------------------------------------------------------------------
print("=" * 70)
print("8.3 BASELINE RESULTS  (price in £; Validation is the comparison set)")
print("=" * 70)
main_cols = ["Model", "Train_R2", "Val_R2", "R2_gap_train_minus_val",
             "Val_MAE", "Val_RMSE", "fit_seconds"]
print(results[main_cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.1f}"))
print("\nSupplementary (relevant to the Deal Advisor's ±5% / ±10% bands) -- Validation:")
sup_cols = ["Model", "Val_MedAPE_pct", "Val_within5_pct", "Val_within10_pct",
            "n_negative_val_predictions", "val_predict_seconds"]
print(results[sup_cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print()

# sanity: موديل R2 بتاعه على Validation أقل من 0 معناه إن في حاجة غلط مش مجرد موديل ضعيف
for _, r in results.iterrows():
    assert r["Val_R2"] > 0, f"{r['Model']}: Validation R2 <= 0 -- something is wrong, investigate before continuing"
neg = results[results["n_negative_val_predictions"] > 0]
for _, r in neg.iterrows():
    print(f"NOTE: {r['Model']} predicted a negative price for {int(r['n_negative_val_predictions'])} "
          f"Validation rows (a car can't cost < £0 -- relevant for the error analysis).")

# ----------------------------------------------------------------------
# 8.4 Save
# ----------------------------------------------------------------------
results.to_csv("baseline_results.csv", index=False)
val_predictions.to_csv("validation_predictions.csv", index=False)

best = results.loc[results["Val_R2"].idxmax()]
print()
print("=" * 70)
print("TRAINING LOG")
print("=" * 70)
print(f"1. 5 baseline models trained on Train only ({X_train.shape[0]} rows x {X_train.shape[1]} columns). "
      "Test set was removed from memory at load time and never used.")
print("2. Same X for every model; target = price in £ (no log transform).")
print(f"3. Highest Validation R2 so far: {best['Model']} ({best['Val_R2']:.4f}). This is a baseline "
      "comparison only -- the final choice also weighs MAE, RMSE, overfitting gap and speed (Parts 9-12).")
print("4. Random Forest uses max_features=0.33 (not 'sqrt') because only 6 of 215 columns are continuous; "
      "Part 11 can compare this on Validation.")
print("5. validation_predictions.csv rows are in the same order as val_fe.csv "
      "(Part 7 verified the index is preserved) -> join by position for the error analysis.")
print()
print(f"Saved: {MODELS_DIR}/*.joblib, baseline_results.csv, validation_predictions.csv")
