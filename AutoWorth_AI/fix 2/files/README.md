# AutoWorth AI: Used Car Price & Smart Deal Advisor

Predicts the expected market price of a used UK car and rates the seller's asking price.

<!-- Add a screenshot of the app: ![App screenshot](docs/screenshot.png) -->

## What it does

Enter a car's details and the seller's asking price. The app shows:

- 💰 the estimated market price
- 🏷️ the seller price
- 💵 the difference in £ and %
- 🚦 a deal rating
- ℹ️ a reliability note for the estimate

## Deal rating

| Asking price vs estimated market price | Rating |
|---|---|
| More than 10% cheaper | 🟢 Great Deal |
| 5–10% cheaper | 🟢 Good Deal |
| Within about ±5% | 🟡 Fair Price |
| 5–10% more expensive | 🟠 Slightly Overpriced |
| More than 10% more expensive | 🔴 Overpriced |

The percentage is rounded to one decimal first, so exactly 5.0% or 10.0% cheaper counts as Good Deal, and exactly 5.0% or 10.0% more expensive counts as Slightly Overpriced.

## How it works

Raw input → validation → feature engineering → preprocessing → Random Forest → deal rating

<!-- Write 2–3 sentences in your own words: dataset, final model, and why you chose it. -->

| Final model (Test set) | Value |
|---|---|
| R² | <!-- from Part 12 --> |
| MAE | <!-- from Part 12 --> |
| RMSE | <!-- from Part 12 --> |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
app.py                    Streamlit interface
deal_advisor.py           validation, prediction and deal rating
part12_final_model.json   which model is the final one
preprocessor.joblib       fitted preprocessing from Part 7
train_fe.csv              training rows with engineered features
models/                   the final model file
requirements.txt
```

The full analysis (cleaning, EDA, models, tuning, evaluation) is in the Colab notebook: <!-- add link -->

## Limitations

<!-- Write 3–5 limitations in your own words (see the limitations list at the end of Part 14). -->
