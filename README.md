# ⚡ EV Route Energy Predictor

> A physics-informed machine learning model that predicts electric vehicle energy consumption based on road topology — slope, surface roughness, bump density, speed, and weather — not just distance.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Motivation

Standard EV range estimators rely almost entirely on distance. Real-world consumption is dramatically affected by factors they ignore:

- A 10° uphill slope can **triple** energy consumption vs flat road
- Cobblestone roads consume **~3× more** than smooth asphalt at the same speed
- Steep downhill segments can **recover energy** through regenerative braking
- Cold weather (< 0°C) adds up to **20% battery drain** from thermal effects

This project builds a model that captures all of these effects.

---

## 🏗️ Project Structure

```
ev-energy-predictor/
│
├── ev_data_generator.py     # Physics-informed synthetic dataset generator
├── EV_Energy_Model.ipynb    # Full training pipeline (Colab-ready)
├── app.py                   # Streamlit demo application
├── requirements.txt
│
├── models/
│   ├── ev_xgboost_model.pkl
│   └── ev_xgboost_model.json
│
└── data/
    ├── ev_segments.csv      # Segment-level dataset (~58k rows)
    └── ev_trips.csv         # Trip-level aggregated dataset (5k trips)
```

---

## ⚙️ Physics Model

Energy for each road segment is computed from four real forces:

```
E = (F_roll + F_gravity + F_drag − E_regen) × temp_factor
```

| Component | Formula | Key Factors |
|---|---|---|
| Rolling Resistance | `Crr × m × g × cos(θ) × d` | Surface type, bump density |
| Gravity | `m × g × sin(θ) × d` | Slope angle, vehicle mass |
| Aerodynamic Drag | `½ × ρ × Cd × A × v² × d` | Speed², frontal area |
| Regenerative Braking | `regen_eff × \|E_gravity\|` | Slope < −1°, vehicle regen rate |
| Temperature Factor | `1 + 0.01 × (20 − T)` for T < 20°C | Cold weather penalty |

Rolling resistance coefficients by surface:

| Surface | Crr | Bump density (bumps/km) |
|---|---|---|
| Smooth Asphalt | 0.008 | 0 – 5 |
| Worn Asphalt | 0.011 | 5 – 20 |
| Cobblestone | 0.018 | 20 – 60 |
| Gravel | 0.025 | 30 – 80 |
| Dirt Road | 0.030 | 40 – 100 |

---

## 📊 Dataset

Since no public dataset captures road topology + EV consumption at segment level, a synthetic dataset was generated using the physics equations above with realistic noise and distributions.

| Property | Value |
|---|---|
| Trips | 5,000 |
| Segments | ~58,000 |
| Features | 24 |
| Target | `consumption_wh_km` |
| Slope distribution | 70% flat, 20% moderate, 10% steep |
| Vehicles | Compact, Sedan, SUV, Pickup |
| Temperature range | −10°C to 45°C |
| Gaussian noise | σ = 2% (sensor/driver variability) |

Negative target values represent net energy recovery on steep downhill segments — this is physically correct and the model learns it.

---

## 🧠 Model Pipeline

```
Raw Segment Data
      │
      ▼
Feature Engineering
  ├── aero_power_proxy  (½ρCdAv³)
  ├── bump_speed_interaction
  ├── slope_mass_interaction
  ├── regen_potential
  ├── temp_deviation
  └── speed_squared
      │
      ▼
 Train / Val / Test Split  (70 / 15 / 15)
      │
      ├──▶ Ridge Baseline
      ├──▶ XGBoost  ◀── GridSearchCV tuning
      └──▶ LightGBM
```

### Results

| Model | MAE (Wh/km) | R² | vs Baseline |
|---|---|---|---|
| Ridge Regression | — | — | — |
| LightGBM | — | — | — |
| **XGBoost (tuned)** | **—** | **—** | **—%** |

> Fill in after running the notebook. Results will vary slightly with random seed.

### SHAP Feature Importance

Top drivers of prediction (from SHAP analysis):
1. `slope_deg` — dominant factor, especially on steep roads
2. `slope_mass_interaction` — heavier vehicles hit harder on hills
3. `aero_power_proxy` — quadratic speed effect at highway speeds
4. `bump_speed_interaction` — rough roads at speed compound significantly
5. `temp_deviation` — cold weather battery penalty

---

## 🖥️ Streamlit App

The demo app lets you build a multi-segment route interactively and see real-time predictions.

**Features:**
- Segment builder with live preview before adding
- Elevation profile visualization
- Energy breakdown per segment (rolling / gravity / drag / regen)
- Cumulative energy + consumption rate chart
- Physics decomposition tab explaining each energy component
- Cold weather warnings

**Run locally:**
```bash
git clone https://github.com/YOUR_USERNAME/ev-energy-predictor
cd ev-energy-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/ev-energy-predictor
cd ev-energy-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python ev_data_generator.py

# 4. Train the model (or use Colab — open EV_Energy_Model.ipynb)
# The notebook saves ev_xgboost_model.pkl automatically

# 5. Run the app
streamlit run app.py
```

---

## 🔮 Future Work

- [ ] **LSTM sequence model** — treat a full trip as a time series of segments for better context-aware prediction
- [ ] **Real data validation** — map OpenStreetMap road segments + elevation API to calibrate synthetic results against real trips
- [ ] **Battery state-of-charge** — model how remaining charge affects regen efficiency
- [ ] **Wind speed** — add headwind/tailwind as a feature (significant at highway speeds)
- [ ] **REST API** — wrap model in FastAPI for integration with navigation apps

---

## 📁 Running on Google Colab

1. Open `EV_Energy_Model.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `ev_segments.csv` when prompted (generated by `ev_data_generator.py`)
3. Run all cells — the notebook handles installs, training, evaluation, and SHAP
4. Download the saved `ev_xgboost_model.pkl` at the end

---

## 📄 License

MIT — free to use, modify, and build on.

---

*Built as part of a 3rd-year AI internship portfolio project.*
