# Quick-Commerce Delivery Intelligence (KLU + PATH3 ML-01)

An end-to-end Machine Learning intelligence engine for quick-commerce food and grocery delivery. The system predicts order delivery time with calibrated uncertainty intervals (reducing MAE from 7.50 min down to 3.12 min, $R^2 = 0.8210$), classifies late-delivery risk at order-placement time (0.9853 ROC-AUC, 0.8844 F1) with cost-optimal thresholding (0.28 vs 0.50 default), maps dark-store expansion priorities via spatial clustering, segments couriers by performance patterns, and delivers interactive predictions through a real-time Streamlit application.

---

## 1. Key System Features & Strategic Innovations

### Key System Features
- **Delivery ETA Prediction Using Machine Learning:** Continuous regression predicting delivery duration with a holdout test MAE of **3.12 minutes** and an $R^2$ of **0.8210**, accompanied by verified 80% coverage prediction intervals ($[-4.70, +5.10]\text{ min}$).
- **Late-Delivery Probability Estimation:** Calibrated posterior probabilities assessing likelihood of exceeding the 30-minute SLA benchmark.
- **Order-Level Risk Classification:** Real-time triage with **0.9853 ROC-AUC** and **0.8844 F1**, tuned to an asymmetric cost-optimal threshold of **0.28** (5:1 penalty model).
- **Geographical Demand-Zone Clustering:** Unsupervised spatial K-Means ($k=4$) isolating under-served geographic delay epicenters.
- **Weather, Traffic, and Temporal Delivery Analysis:** Deep multidimensional breakdown of dinner surges (19:00–22:00, 55.4% late rate), traffic friction multipliers (+14 min), and atmospheric friction.
- **Model Explainability Through Permutation Feature Importance:** Interpretable diagnostic ranking identifying `Road_traffic_density`, `multiple_deliveries`, `distance_km`, and `Weatherconditions` as the top causal drivers.
- **Operational Risk Summaries & Recommendations:** Actionable ROI-driven strategies targeting real-time priority dispatching, micro-fulfillment dark stores, and dynamic surge rostering.
- **Interactive Streamlit Dashboard Prototype:** Production-grade 6-tab operational interface featuring real-time triage, what-if planning simulators, model benchmarks, and cluster maps.

### Strategic Innovations
- **Dual-Model Intelligence:** Combines exact continuous ETA regression with cost-calibrated late risk classification in a single joint-inference pass at checkout.
- **Risk-Aware Order Prioritization:** Triages incoming orders dynamically into Priority 1 (Urgent/At-Risk couriers), Priority 2 (Standard), and Priority 3 (Batchable for pooled routing).
- **Geographic Demand & Delivery-Performance Intelligence:** Combines spatial K-Means clustering with delay severity metrics to map network bottlenecks and dark-store candidates (Zone 3: 39.12% late rate).
- **Explainable Machine-Learning Predictions:** SHAP factor attributions detailing the exact marginal delay contribution of traffic, weather, distance, and batching.
- **Foundation for Rider-Allocation Simulation & Dark-Store Planning:** Interactive what-if planning simulator to model the exact operational ROI of adding fleet surge capacity and opening micro-fulfillment hubs.

---

## 2. Setup
```bash
# Clone the repository
git clone <repo-url>
cd Manthan-Hackathon

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Raw dataset is located at `data/raw/train.csv` and configured in `config.yaml`.

---

## 3. How to Run
- **Execute Full Pipeline (Data Cleaning → Models → Clustering → Explainability → Reports):**
  ```bash
  python run_all.py
  ```
- **Launch Interactive Streamlit Demo App:**
  ```bash
  streamlit run app/streamlit_app.py
  ```

---

## 4. Folder Structure
```
Manthan-Hackathon/
├── app/
│   └── streamlit_app.py           # Interactive multi-tab web application
├── config.yaml                    # Single source of truth (SLA, cost ratios, seeds)
├── data/
│   ├── raw/train.csv              # Organiser raw dataset (45,593 rows)
│   └── processed/clean.parquet    # Cleaned dataset (41,497 rows)
├── models/
│   ├── regressor.joblib           # Trained Best Regressor (Gradient Boosting)
│   ├── classifier.joblib          # Trained Best Classifier (Gradient Boosting)
│   ├── metadata_regression.json   # Regressor metrics & ETA band parameters
│   └── metadata_classifier.json   # Classifier metrics & optimal threshold
├── notebooks/                     # Analytical narrative notebooks
├── reports/
│   ├── data_dictionary.md         # Exhaustive column schemas & preprocessing rules
│   └── executive_summary.md       # Comprehensive business intelligence report
├── results/
│   ├── figures/                   # Diagnostic plots & SHAP explainability charts
│   ├── cleaning_log.csv           # Audit trail of every preprocessing transformation
│   ├── regression_comparison.csv  # 4-model regression benchmark table
│   ├── classification_comparison.csv # 4-model classification benchmark table
│   ├── threshold_sensitivity.csv  # Cost-ratio sensitivity analysis
│   ├── zones_table.csv            # Dark-store geographic cluster scorecard
│   ├── rider_segments.csv         # Courier behavioral segment scorecard
│   ├── surge_windows.csv          # 33 operational surge bottleneck windows
│   └── recommendations.md         # 3 prioritized ROI-driven interventions
├── src/
│   ├── cleaning.py                # Data quality, parsing & cleaning pipeline
│   ├── features.py                # Zero-leakage feature engineering & ColumnTransformer
│   ├── regression.py              # Regressor training & uncertainty intervals
│   ├── classification.py          # Classifier training & threshold optimization
│   └── clustering.py              # Dark-store zones & courier segmentation
├── run_all.py                     # Master one-click reproduction script
└── TEAM_LOG.md                    # Team contributions & task distribution
```

---

## 5. Key Definitions & Methodological Assumptions
- **Late Delivery:** Order duration exceeding candidate SLA of **30 minutes**.
- **Distance Calculation:** Spherical Haversine formula ($R = 6,371\text{ km}$) computed between merchant and customer coordinates.
- **Zero-Leakage Policy:** Features are strictly limited to signals available at order placement. `Time_Order_picked` and derived pickup delay statistics are strictly eliminated.
- **Asymmetric Operational Cost Ratio:** Cost of a missed late delivery (customer churn, refund, support ticket) is assumed at **5 units**, while proactive intervention (priority dispatch, proactive SMS alert) is **1 unit** ($5:1$ ratio).
- **ETA Uncertainty Coverage:** Computed from residual quantiles $[Q_{0.10}, Q_{0.90}]$ on validation residuals, validating an empirical **80.0% coverage** on holdout test data.

---

## 6. Results Summary

### 6.1. Regression Benchmark (Holdout Test Set: 8,300 Orders)
| Model | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Validation MAE (min) |
|---|---|---|---|---|
| **Baseline (Mean)** | 7.5010 | 9.2792 | -0.0001 | 7.6455 |
| **Ridge Regression** | 4.7712 | 5.9734 | 0.5855 | 4.7601 |
| **Random Forest** | 3.1221 | 3.9421 | 0.8195 | 3.1695 |
| **Gradient Boosting (Best)** | **3.1214** | **3.9261** | **0.8210** | **3.1456** |

- **ETA Uncertainty Interval:** $[-4.70, +5.10]\text{ min}$ around predicted ETA.
- **Empirical Test Coverage:** **80.0%** (target: 80%).

### 6.2. Classification Benchmark (Holdout Test Set: 8,300 Orders)
| Model | Test ROC-AUC | Test PR-AUC | Test F1-Score | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **Baseline (Stratified)** | 0.5004 | 0.2927 | 0.2913 | 0.2930 | 0.2895 |
| **Logistic Regression** | 0.9373 | 0.8871 | 0.7739 | 0.7162 | 0.8418 |
| **Random Forest** | 0.9833 | 0.9665 | 0.8770 | 0.8659 | 0.8884 |
| **Gradient Boosting (Best)** | **0.9853** | **0.9703** | **0.8844** | **0.8640** | **0.9057** |

- **Optimal Decision Threshold:** **0.28** (vs default 0.50), minimizing expected business loss.

### 6.3. Dark-Store Zones & Courier Segments (ALL 4 Discovered Zones)
- **All Discovered Clusters ($k=4$):**
  - **Zone 3 (Rank #1 Expansion Target):** 62 stores, 4,469 orders, **37.95% late rate** (+7.97 pp excess delay), Center: `(22.548°N, 77.428°E)`.
  - **Zone 0 (East Region Hub):** 35 stores, 3,043 orders, 29.58% late rate, Center: `(22.978°N, 86.882°E)`.
  - **Zone 1 (South & West Mega Hub):** 219 stores, 29,311 orders, 28.86% late rate, Center: `(16.456°N, 75.859°E)` (Covers Maharashtra, Karnataka, Telangana, Tamil Nadu, Andhra Pradesh).
  - **Zone 2 (North & West Hub):** 72 stores, 4,674 orders, 29.59% late rate, Center: `(27.437°N, 77.968°E)`.
- **Spatial Exclusion Audit:** 3,640 records with Atlantic Ocean `(0, 0)` sentinels and 431 distance outliers (>100 km) purged; 41,497 orders across 388 stores clustered.
- **Enterprise Folium Map:** Exported to `results/delivery_intelligence_map.html` and embedded in Streamlit Tab 4 with real OSRM road routes.
- **Courier Segments:**
  - *Segment 1 (High-Volume Core Couriers):* 615 couriers, 24.66% late rate, 25.25 min avg duration.
  - *Segment 0 (Long-Distance Specialists):* 481 couriers, 37.60% late rate, 27.81 min avg duration.

---

## 7. Business Recommendations

1. **Deploy Order-Placement Risk Scoring (Optimal Threshold 0.28):** Flag high-risk orders instantly upon submission. Trigger priority courier assignment and dispatch proactive notifications ("Your order ETA is 26 min [range: 21–31 min]").
2. **Establish Micro-Fulfillment Dark Store at Zone 3:** Open a micro-hub near `(22.548°N, 77.428°E)` to compress delivery radiuses in the network's most delayed sector. A 30% distance reduction is estimated to recover ~1.5–2.0 minutes per trip.
3. **Dynamic Surge Rostering for 33 Bottleneck Windows:** Deploy 25–30% additional fleet capacity during peak dinner windows (Wednesday & Friday 20:00–21:00) where late rates exceed 53%.

---

## 8. Limitations & Boundary Conditions
1. **Haversine vs True Road Network:** Distance is measured as great-circle distance; actual road transit distance may vary with urban topography and physical obstacles.
2. **Candidate SLA:** Baseline SLA is set to 30 minutes. If the organizer publishes an alternate SLA (e.g. 25 or 35 min), models can be re-calibrated via `config.yaml`.
3. **Correlational Projections:** Estimated savings are based on observed observational distributions; production deployment should be accompanied by staged A/B experimentation.

---

## 9. Team Contributions & Role Breakdown
| Role | Core Owner | Responsibilities & Completed Deliverables |
|---|---|---|
| **P1 — Data & Insights Lead** | P1 | Data cleaning pipeline (`src/cleaning.py`), 4,096 invalid records cleaned, zero-leakage feature engineering (`src/features.py`), 6 EDA figures, data dictionary (`reports/data_dictionary.md`). |
| **P2 — Regression & App Lead** | P2 | 4-tier regression models (`src/regression.py`), 3.12 min test MAE, 80% coverage ETA uncertainty intervals, out-of-fold predictions, interactive Streamlit demo (`app/streamlit_app.py`). |
| **P3 — Classification & Risk Lead** | P3 | 4-tier classification models (`src/classification.py`), 0.9853 test ROC-AUC, 5:1 asymmetric cost-sensitive threshold optimization (0.28), 33 surge bottleneck windows. |
| **P4 — Clustering & Business Lead** | P4 | Spatial dark-store clustering ($k=4$, `src/clustering.py`), courier segmentation ($k=2$), 3 ROI-driven recommendations (`results/recommendations.md`), executive report (`reports/executive_summary.md`), master orchestration (`run_all.py`). |
