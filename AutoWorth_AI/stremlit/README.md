# AutoWorth AI — Streamlit App

**Live demo:** https://autoworthai-hy2xhgsfunvewy7kwmbhdw.streamlit.app/  
**Repository:** https://github.com/ALII0101/AutoWorth_AI

This folder contains the Streamlit user interface and the prediction/deal-advisor logic for AutoWorth AI. It loads a saved Random Forest model and fitted preprocessing object; it does not train or tune a model when the app starts.

## Features

- Accepts make, model, model year, mileage, transmission, fuel type, engine size, and seller asking price.
- Allows MPG and annual tax to be entered as optional details.
- Estimates a used-car market price in GBP.
- Compares the asking price with the estimate and displays a deal rating.
- Displays reliability warnings and explains when optional fields have been imputed.

## Run locally

Run Git LFS from the repository root first, because the Random Forest artifact is stored in LFS:

```bash
git lfs install
git lfs pull
```

Then, from this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit interface.
- `deal_advisor.py` — input validation, feature construction, prediction, reliability notes, and deal-rating logic.
- `part12_final_model.json` — final model selection metadata and evaluation results.
- `random_forest.joblib` — saved Random Forest model (Git LFS).
- `preprocessor.joblib` — fitted preprocessing transformer.
- `train_fe.csv` — training data used for supported categories, ranges, and similar-car median lookups.
- `test_advisor.py` — application checks.
- `requirements.txt` — app runtime dependencies.

## Model and rating behaviour

The deployed model is a Random Forest trained on historical UK used-car listings, with data ending at model year 2020. The app only accepts make/model combinations represented in its training data and does not value model years newer than 2020.

The rating compares the seller's asking price with the model's estimated price. It does not inspect vehicle condition, accident history, service history, location, trim, or optional extras. Treat it as an initial price-screening signal, not a purchase recommendation or a guaranteed valuation.

## Checks

After installing the dependencies and retrieving the model artifact, run:

```bash
python test_advisor.py
```

The test script checks dependency/model compatibility, prediction parity, feature construction, rating thresholds, invalid inputs, and expected behaviour for representative vehicles.
