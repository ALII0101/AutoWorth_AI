# =====================================================================
# PART 16.0: Setup, settings and Google Drive
# =====================================================================
# Where the files are:
#   Parts 7-12 saved their files with relative paths, i.e. into the Colab session folder (/content),
#   NOT into Drive. This part therefore looks for its INPUT files in Drive first, then in /content.
#   Everything this part CREATES (tuning table, .keras models, results) goes to Drive (OUT_DIR),
#   so a runtime reset cannot delete hours of training.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json, time, datetime, shutil
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sklearn
import keras
from keras import layers
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DRIVE_MOUNTED = False
try:                                                   # Colab: mount Drive
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/AutoWorth_AI"   # <-- change to your project folder on Drive
    DRIVE_MOUNTED = True
except ImportError:                                    # running locally
    PROJECT_DIR = "."

INPUT_DATA = "preprocessed_data.joblib"                # Part 7: X_train/X_val/X_test + y_train/y_val/y_test
DECISION_FILE = "part12_final_model.json"              # Part 12: final model + its Validation/Test numbers + noise rule
OUT_DIR = os.path.join(PROJECT_DIR, "part16")          # everything this part saves (on Drive in Colab)
os.makedirs(OUT_DIR, exist_ok=True)

BACKUP_INPUTS_TO_DRIVE = True                          # copy the 3 input files from /content to Drive (once), so a reset cannot lose them
EXPECTED_TRAIN_ROWS = 68_373                           # Part 7 Train rows
EXPECTED_FEATURES = 215                                # columns after the Part 7 preprocessor
CHECK_AGAINST_PART12 = True                            # RF on Validation must reproduce the Part 12 numbers

SEEDS = [42, 123, 456, 789, 2026]                      # 5 runs of the chosen configuration
TUNE_SEED = 42                                         # one seed per tuning trial
MAX_EPOCHS, ES_PATIENCE = 150, 15
EVALUATE_TEST_NOW = False                              # 1st run: False.  After reviewing Validation: True, run the cell again

# The tuning budget is fixed HERE, before anything is trained: 8 trials.
# Selection rule (also fixed now): the trial with the lowest Validation MAE in pounds.
TRIALS = {
    "T1": dict(units=[128, 64, 32],   dropout=0.2, lr=1e-3, batch=256),
    "T2": dict(units=[256, 128, 64],  dropout=0.2, lr=1e-3, batch=256),
    "T3": dict(units=[128, 64, 32],   dropout=0.1, lr=1e-3, batch=256),
    "T4": dict(units=[256, 128, 64],  dropout=0.3, lr=1e-3, batch=512),
    "T5": dict(units=[128, 64, 32],   dropout=0.2, lr=3e-4, batch=256),
    "T6": dict(units=[512, 256, 128], dropout=0.3, lr=1e-3, batch=512),
    "T7": dict(units=[64, 32],        dropout=0.1, lr=1e-3, batch=256),
    "T8": dict(units=[256, 128, 64],  dropout=0.2, lr=3e-3, batch=512),
}

print(f"scikit-learn {sklearn.__version__} | keras {keras.__version__} | backend {keras.backend.backend()}")
if sklearn.__version__ != "1.6.1":
    print("NOTE: the saved .joblib files were written with scikit-learn 1.6.1. If loading fails below, "
          "run  pip install scikit-learn==1.6.1  and restart the runtime.")
print(f"Project folder (outputs): {os.path.abspath(PROJECT_DIR)}")


# =====================================================================
# 16.1  Find, load and validate the Part 7 data (nothing is recomputed)
# =====================================================================
def find_input_dir(required):
    """First folder (Drive project folder, current folder, /content) that holds ALL required files."""
    searched = []
    for d in (PROJECT_DIR, os.getcwd(), "/content", "."):
        d = os.path.abspath(d)
        if d not in searched:
            searched.append(d)
    for d in searched:
        if all(os.path.exists(os.path.join(d, f)) for f in required):
            return d, searched
    return None, searched

INPUT_DIR, searched_dirs = find_input_dir([INPUT_DATA, DECISION_FILE])
assert INPUT_DIR is not None, (
    f"{INPUT_DATA} and {DECISION_FILE} were not found together in any of: {searched_dirs}. "
    "Run Parts 7 and 12 first (or copy both files into one of these folders).")
print(f"Input files found in: {INPUT_DIR}")

data = joblib.load(os.path.join(INPUT_DIR, INPUT_DATA))
with open(os.path.join(INPUT_DIR, DECISION_FILE), encoding="utf-8") as f:
    decision = json.load(f)

# the model file named in the decision (e.g. models/random_forest.joblib), same folder search
model_rel = decision["model_file"]
model_dirs = list(dict.fromkeys([INPUT_DIR] + searched_dirs))      # INPUT_DIR first, no duplicates
rf_path = model_rel if os.path.isabs(model_rel) else next(
    (os.path.join(d, model_rel) for d in model_dirs if os.path.exists(os.path.join(d, model_rel))), None)
assert rf_path is not None and os.path.exists(rf_path), (
    f"{model_rel} (the model named in {DECISION_FILE}) was not found in: {model_dirs}")
print(f"Part 12 model file   : {rf_path}")

if BACKUP_INPUTS_TO_DRIVE and DRIVE_MOUNTED:
    for src, rel in [(os.path.join(INPUT_DIR, INPUT_DATA), INPUT_DATA),
                     (os.path.join(INPUT_DIR, DECISION_FILE), DECISION_FILE),
                     (rf_path, model_rel)]:
        dst = os.path.join(PROJECT_DIR, rel)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Backed up to Drive: {rel}")

def as_series(y, X):
    """Part 16 needs y as a pandas Series aligned with X (an array is wrapped with X's index)."""
    return y if isinstance(y, pd.Series) else pd.Series(np.asarray(y), index=X.index)

X_train, X_val = data["X_train"], data["X_val"]
y_train, y_val = as_series(data["y_train"], X_train), as_series(data["y_val"], X_val)
# Test stays inside `data` and is NOT used until section 16.7 (only its column names are checked here).

assert X_train.shape == (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURES), f"unexpected Train shape {X_train.shape}"
assert list(X_val.columns) == list(X_train.columns) == list(data["X_test"].columns), "columns differ between splits"
assert len(X_val) == len(y_val) and len(X_train) == len(y_train)
assert X_train.index.equals(y_train.index) and X_val.index.equals(y_val.index), "X / y rows are not aligned"
assert not X_train.isnull().any().any() and not X_val.isnull().any().any(), "NaN found in the processed features"
assert not any("price" in c for c in X_train.columns), "the target is among the features"
assert (y_train > 0).all() and (y_val > 0).all(), "log1p needs positive prices"

Xtr = X_train.to_numpy("float32")
Xva = X_val.to_numpy("float32")
print(f"Train: {X_train.shape} | Validation: {X_val.shape} | Test: {data['X_test'].shape} (not used yet)")
print(f"{EXPECTED_FEATURES} features, same columns and order in all three splits, no NaN, price not among the features.")


# =====================================================================
# 16.2  Metrics + the Random Forest reference on Validation (inference only, no retraining)
# =====================================================================
METRIC_KEYS = ["R2", "MAE", "RMSE", "MedAPE", "within5", "within10"]

def metrics(y_true, y_pred):
    """All errors in pounds / percent of the REAL price (never in log units)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    ape = np.abs(y_pred - y_true) / y_true * 100                 # absolute % error per car
    return {"R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MedAPE": float(np.median(ape)),
            "within5": float((ape <= 5).mean() * 100),
            "within10": float((ape <= 10).mean() * 100)}

rf = joblib.load(rf_path)
rf_val_pred = np.asarray(rf.predict(X_val), dtype="float64")
rf_val = metrics(y_val, rf_val_pred)
del rf                                                            # free ~120 MB; only its predictions are needed

if CHECK_AGAINST_PART12:
    # R2 / MAE / RMSE are the only Validation numbers stored in part12_final_model.json.
    for k in ["R2", "MAE", "RMSE"]:
        ref = decision["validation"][k]
        assert np.isclose(rf_val[k], ref, rtol=1e-4), f"RF Validation {k}: got {rf_val[k]:.6f} but Part 12 says {ref:.6f}"
    print("OK: the RF Validation R2 / MAE / RMSE computed here match Part 12 (metric code + data alignment verified).")

print("\nRandom Forest on Validation (Part 12 model, not retrained):")
print(pd.Series(rf_val).loc[METRIC_KEYS].round(4).to_string())


# =====================================================================
# 16.3  Target = log1p(price), and the network / training helpers
# =====================================================================
ytr_log = np.log1p(y_train.to_numpy("float64"))
yva_log = np.log1p(y_val.to_numpy("float64"))
assert np.allclose(np.expm1(yva_log), y_val.to_numpy("float64")), "log1p / expm1 round trip failed"
print(f"price skew: {y_train.skew():.2f}  ->  log1p(price) skew: {pd.Series(ytr_log).skew():.2f}")
ytr_log = ytr_log.astype("float32")
yva_log = yva_log.astype("float32")

def build_model(n_features, units, dropout, lr):
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for i, u in enumerate(units):
        x = layers.Dense(u, activation="relu")(x)
        if dropout > 0 and i < len(units) - 1:                    # no dropout right before the output layer
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)                                  # linear output = predicted log1p(price)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def train_one(cfg, seed):
    keras.utils.set_random_seed(seed)                             # python + numpy + backend seeds
    model = build_model(Xtr.shape[1], cfg["units"], cfg["dropout"], cfg["lr"])
    callbacks = [
        # NOTE: Validation drives early stopping, so Validation scores are slightly optimistic. Test is untouched.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    hist = model.fit(Xtr, ytr_log, validation_data=(Xva, yva_log), epochs=MAX_EPOCHS,
                     batch_size=cfg["batch"], callbacks=callbacks, verbose=0)
    return model, hist.history

def predict_price(model, X):
    log_pred = model.predict(X, batch_size=2048, verbose=0).ravel().astype("float64")
    return np.expm1(log_pred)                                     # back to pounds BEFORE any metric

# Seeds make runs repeatable, but GPU kernels can still differ slightly between sessions.


# =====================================================================
# 16.4  Tuning: the 8 declared trials, ONE seed each, judged on Validation only
# =====================================================================
TUNE_CSV = os.path.join(OUT_DIR, "part16_tuning.csv")
rows = pd.read_csv(TUNE_CSV).to_dict("records") if os.path.exists(TUNE_CSV) else []
for r in rows:                                                    # a checkpoint is valid only for the SAME trial definitions
    assert r["trial"] in TRIALS and r["config"] == json.dumps(TRIALS[r["trial"]], sort_keys=True), (
        "TRIALS changed since the saved tuning file: delete part16_tuning.csv and the nn_*.keras files, then re-run")
done = {r["trial"] for r in rows}

for tid, cfg in TRIALS.items():
    if tid in done:
        print(f"{tid}: loaded from checkpoint")
        continue
    t0 = time.time()
    model, hist = train_one(cfg, TUNE_SEED)
    m = metrics(y_val, predict_price(model, Xva))
    rows.append({"trial": tid, "config": json.dumps(cfg, sort_keys=True), "epochs": len(hist["loss"]), **m,
                 "minutes": round((time.time() - t0) / 60, 1)})
    pd.DataFrame(rows).to_csv(TUNE_CSV, index=False)              # checkpoint after every trial
    print(f"{tid}: {len(hist['loss'])} epochs | Val MAE £{m['MAE']:,.0f} | R2 {m['R2']:.4f} | {rows[-1]['minutes']} min")
    keras.backend.clear_session()

tuning = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
BEST_ID = str(tuning.loc[0, "trial"])
BEST_CFG = TRIALS[BEST_ID]
print("\nTuning results (Validation, sorted by MAE):")
print(tuning[["trial", "config", "epochs", "R2", "MAE", "RMSE", "within10"]].round(4).to_string(index=False))
print(f"\nSelected by the pre-declared rule (lowest Validation MAE): {BEST_ID} -> {BEST_CFG}")


# =====================================================================
# 16.5  The chosen configuration with 5 seeds + comparison with the Random Forest (Validation)
# =====================================================================
final_models, histories, val_preds, seed_rows = {}, {}, {}, []
for seed in SEEDS:
    path = os.path.join(OUT_DIR, f"nn_{BEST_ID}_seed{seed}.keras")
    hist_path = path[:-len(".keras")] + "_history.json"
    if os.path.exists(path) and os.path.exists(hist_path):
        model = keras.models.load_model(path)
        with open(hist_path) as f:
            hist = json.load(f)
        source = "checkpoint"
    else:
        model, hist = train_one(BEST_CFG, seed)
        model.save(path)
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in hist.items()}, f)
        source = "trained"
    final_models[seed], histories[seed] = model, hist
    val_preds[seed] = predict_price(model, Xva)
    seed_rows.append({"seed": seed, **metrics(y_val, val_preds[seed])})
    print(f"seed {seed}: {source} | Val MAE £{seed_rows[-1]['MAE']:,.0f} | R2 {seed_rows[-1]['R2']:.4f}")
    assert val_preds[seed].min() > 0, "a non-positive price was predicted"

seed_df = pd.DataFrame(seed_rows).set_index("seed")
summary = pd.DataFrame({"Random Forest (Part 12)": pd.Series(rf_val),
                        "NN mean": seed_df.mean(), "NN std": seed_df.std()}).loc[METRIC_KEYS]
print("\nValidation: Random Forest vs Neural Network (5 seeds, std = sample std):")
print(summary.round(4).to_string())

# Verdict on Validation, with the SAME noise levels Part 12 used (R2 0.002, MAE £15).
R2_NOISE, MAE_NOISE = decision["rule"]["R2_NOISE"], decision["rule"]["MAE_NOISE_gbp"]
d_r2 = seed_df["R2"].mean() - rf_val["R2"]                       # > 0 -> NN better
d_mae = rf_val["MAE"] - seed_df["MAE"].mean()                    # > 0 -> NN better (lower error)
if d_r2 > R2_NOISE and d_mae > MAE_NOISE:
    VERDICT = "NN clearly better on Validation"
elif d_r2 < -R2_NOISE and d_mae < -MAE_NOISE:
    VERDICT = "RF clearly better on Validation"
else:
    VERDICT = "no clear difference on Validation (inside the Part 12 noise levels, or the two metrics disagree)"
print(f"\nNN minus RF: R2 {d_r2:+.4f} (noise level {R2_NOISE}) | MAE gain of the NN £{d_mae:+,.0f} "
      f"(positive = NN has the lower error; noise level £{MAE_NOISE:g})")
print(f"VERDICT: {VERDICT}")


# =====================================================================
# 16.6  Training curves + error by price band (Validation only)
# =====================================================================
first = SEEDS[0]
plt.figure(figsize=(7, 4))
plt.plot(histories[first]["loss"], label="Train loss")
plt.plot(histories[first]["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("MSE on log1p(price)")
plt.title(f"Training curves ({BEST_ID}, seed {first})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part16_training_curves.png"), dpi=130)
plt.show()

bins = [0, 5_000, 10_000, 20_000, 40_000, np.inf]
labels = ["<£5k", "£5-10k", "£10-20k", "£20-40k", ">=£40k"]
band = pd.cut(y_val.to_numpy(), bins=bins, labels=labels, right=False)
y_true = y_val.to_numpy("float64")
table = []
for lab in labels:
    m = np.asarray(band == lab)
    if m.sum() == 0:
        continue
    nn_mae = np.mean([mean_absolute_error(y_true[m], val_preds[s][m]) for s in SEEDS])
    nn_ape = np.mean([np.median(np.abs(val_preds[s][m] - y_true[m]) / y_true[m] * 100) for s in SEEDS])
    rf_ape = np.median(np.abs(rf_val_pred[m] - y_true[m]) / y_true[m] * 100)
    table.append({"price band": lab, "cars": int(m.sum()),
                  "RF MAE £": mean_absolute_error(y_true[m], rf_val_pred[m]), "NN MAE £": nn_mae,
                  "RF MedAPE %": rf_ape, "NN MedAPE %": nn_ape})
print("Validation error by TRUE price band (NN = average over the 5 seeds):")
print(pd.DataFrame(table).round(2).to_string(index=False))


# =====================================================================
# 16.7  Test, evaluated ONCE on the frozen models
# =====================================================================
RESULT_JSON = os.path.join(OUT_DIR, "part16_test_results.json")
res = None
if os.path.exists(RESULT_JSON):
    with open(RESULT_JSON, encoding="utf-8") as f:
        res = json.load(f)
    print("Test was already evaluated -> showing the saved result (NOT evaluating again).")
elif not EVALUATE_TEST_NOW:
    print("Test NOT evaluated. When Validation work is finished, set EVALUATE_TEST_NOW = True in the settings at the "
          "top of this cell and run it again (nothing is retrained: the models load from Drive).")
else:
    # the Part 12 Random Forest Test numbers are needed for the comparison table: check BEFORE touching Test
    assert decision.get("test"), (
        f"{DECISION_FILE} contains no Test numbers for the Random Forest. Run Part 12 (the updated version) once "
        "so that it fills them in, then run this cell again.")
    Xte = data["X_test"].to_numpy("float32")
    yte = data["y_test"]
    assert Xte.shape[1] == EXPECTED_FEATURES and len(Xte) == len(yte)
    per_seed = {}
    for seed in SEEDS:
        pred = predict_price(final_models[seed], Xte)
        assert pred.min() > 0, "a non-positive price was predicted"
        per_seed[str(seed)] = metrics(yte, pred)
    test_df = pd.DataFrame(per_seed).T
    res = {"best_trial": BEST_ID, "config": BEST_CFG, "seeds": SEEDS, "validation_verdict": VERDICT,
           "per_seed": per_seed, "mean": test_df.mean().to_dict(), "std": test_df.std().to_dict(),
           "rf_test_part12": {k: decision["test"][k] for k in METRIC_KEYS},
           "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Test evaluated once and saved to part16_test_results.json")

if res is not None:
    final = pd.DataFrame({"Random Forest (Part 12)": pd.Series(res["rf_test_part12"]),
                          "NN mean": pd.Series(res["mean"]), "NN std": pd.Series(res["std"])}).loc[METRIC_KEYS]
    print(f"\nTEST: Random Forest vs Neural Network ({res['best_trial']}, {len(res['seeds'])} seeds)")
    print(final.round(4).to_string())
    print(f"\nDecision rule: the model choice was made on Validation -> {res['validation_verdict']}.")
    print("Test numbers are the final unbiased estimate; they do not change that choice.")
