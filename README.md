# 🏭 Exploratory Data Analysis & Anomaly Detection – PiSA Pharmaceuticals

This project presents an Exploratory Data Analysis (EDA) and anomaly detection pipeline on industrial sensor data from the **bottle-blowing machine** used in the production of *Electrolit*, the flagship product of **PiSA Pharmaceuticals**.

The goal is to identify patterns of abnormal temperature readings that may be shortening the machine's lifespan from a full year to just 2–3 months.

---

## 🧪 Context

PiSA is a major pharmaceutical manufacturer in Mexico with 14 production plants and 1,500+ product lines. The Electrolit manufacturing process relies heavily on a bottle-blowing machine with **100+ sensors** that track:
- Temperature
- Pressure
- Current
- Angular speed
- Machine speed

---

## 🧠 Hypothesis

High temperature variability recorded by sensors may be causing **thermal stress** on machine components, reducing the equipment's lifespan.

---

## 🧰 Methodology

### 📁 Data

- Two `.parquet` files (~1.8GB each), corresponding to November 2024 and January 2025.
- ~45 million records in total.
- Key columns: `user_ts`, `variable`, `valor`, `mensaje`.

### ⚙️ Data Processing

- Processed in Python using `pandas` with chunking (100,000 rows per chunk).
- Transformed into pivot tables with one column per temperature variable.
- Exported to daily `.csv` files for EDA.

### 📊 Variable Selection

- Applied a missing value filter (80th percentile) to keep the most complete variables.
- Selected 10–13 key temperature variables per day, including:
  - `energyPerPreform_CurrentPreform NeckFinishTemperature.0`
  - `powerPerPreform_CurrentPreform TemperatureOvenInfeed.0`
  - `value_ActualTemperatureCoolingCircuit2.0`, etc.

### 🚨 Anomaly Detection

- Used `KNN` model from **PyOD** library:
  ```python
  n_neighbors = 20
  contamination = 0.0025
