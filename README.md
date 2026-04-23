# 🍿 Cinema Audience Forecasting using Time-Series Data

## 📋 Project Overview

This project presents a comprehensive approach to forecasting daily cinema audience attendance. Leveraging time-series data from two distinct platforms—**BookNow** (an online booking system) and **CinePOS** (an in-theatre point-of-sale system)—the goal is to predict daily visitor counts for 826 specific theatres across a 53-day forecast window.

Accurate demand forecasts allow cinema managers to optimize staffing schedules, resource allocation, concession inventory, and ticket pricing.

---

## 📂 Dataset Architecture

The project integrates 8 interrelated datasets encompassing temporal patterns, geographical data, and aggregate booking behaviors:
* **Booking Records:** Over 1.6 million `CinePOS` bookings and ~68,000 `BookNow` online bookings.
* **Theatre Metadata:** Mapped 150 BookNow theatres to their offline CinePOS counterparts using relation mapping.
* **Target Variable:** Captured daily audience counts spanning 214,046 total historic visit records (Training Window: *Jan 1, 2023 - Feb 28, 2024*).
* Dataset Link:https://www.kaggle.com/competitions/Cinema_Audience_Forecasting_challenge/data

---

## 🔍 Core Workflow & Methodology

### 1. Data Preprocessing & Aggregation
* Standardized `show_datetime` and `booking_datetime` strings to strict date formats.
* Grouped massive historical ticketing data into unified daily blocks by `theater_id`.
* Executed robust hierarchical missing-value handling. Missing geographic/string fields were bridged using cross-platform (CinePOS/BookNow) fallback merges, followed by modal imputations. 

### 2. Feature Engineering
The pipeline transformed raw dates and identifiers into 13 high-value ML features:
* **Temporal Indicators:** Extracted `is_weekday`, `year`, `month`, `date`, `daynum`, and `week`.
* **Lag Features (Time-Series):** Automatically generated `lag_1`, `lag_2`, and `lag_7` (same day of the previous week) audience shifts using grouped historical concatenations.
* **Custom Target Encoding:** Developed a custom Scikit-Learn transformer class (`CustomEncoder`) that maps theatre IDs to their historic mean audience count, giving trees direct numeric magnitude rather than dealing with massively sparse One-Hot geometries.

### 3. Model Architecture & Tuning
Four gradient-boosted branches were trained across parallel pipelines (via `ColumnTransformer`), optimized using **RandomizedSearchCV** (50 iterations, 5-fold CV):
1. **Gradient Boosting Regressor (GBR)**
2. **XGBoost Regressor (XGB)** — *(Best Params: Learning Rate 0.01, 350 Estimators, Depth 10)*
3. **LightGBM Regressor (LGBM)**
4. **Voting Regressor (Ensemble)** — Modeled as an aggregated average of the three primary trees.

---

## 📈 Model Performance Results

The models were evaluated heavily on a held-out test block (data succeeding Jan 31, 2024).

| Model | R² Score | Mean Squared Error (MSE) |
|-------|----------|--------------------------|
| **XGBoost** | **0.5556** | **424.58** |
| Voting Ensemble | 0.5319 | 447.23 |
| LightGBM | 0.5273 | 451.59 |
| Gradient Boosting | 0.5111 | 467.05 |

> 🏆 **Leaderboard Context:** For authenticity and comparison, the top competitor on the leaderboard achieved an R² score of around **0.5568**, while my submission scored **0.506**. The standalone XGBoost model achieved highly competitive baselines completely independently.

### Post-Validation Refit
After selecting hyperparameters, the final models were refitted continuously onto the *complete* available historic dataset.
* **XGBoost Overall Final R²:** `0.6031` (MSE: `427.92`)
* This proved the architecture's capacity to continue scaling predictive logic gracefully with increased data throughput.

---

## 💡 Key Observations

* **Autocorrelation is King:** The `lag_1` and `lag_7` historical offsets were massively impactful predictive markers, proving standard cyclic behavioral dependencies in entertainment attendance.
* **Target Encoding Success:** Utilizing target encoding for individual theatre profiles structurally improved the tree splits over standard categorical One-Hot expansion.
* **Ensemble Limitations:** In this exact dimensional space, the Voting Regressor did not mathematically outperform the standalone XGBoost pipeline, demonstrating XGBoost had naturally saturated the available temporal variance mappings.

---

## 🚀 Conclusion & Future Scope

The project confidently demonstrates the viability of utilizing lagged time-series structures combined with gradient-boosted machines (XGBoost) to resolve complex, real-world cinema footprint forecasts. 

**Future Pipeline Upgrades:**
Integrating advanced, long-range dependency extraction architectures (like LSTM or Transformers) and concatenating physical holiday/movie-release-calendar datasets could potentially yield further performance improvements.


