# Executive Summary & Business Intelligence Report
## Quick-Commerce Delivery Intelligence (KLU + PATH3 ML-01)

**Date:** October 31, 2026  
**Audience:** Executive Committee, Logistics Operations, and Data Science Evaluation Panel  
**Dataset Scale:** 45,593 raw records → 41,497 audit-verified clean orders  

---

## 1. Executive Summary & Headline Result

This project delivers an end-to-end Machine Learning intelligence engine for quick-commerce delivery optimization. Operating exclusively on data available at order-placement time (strictly eliminating target leakage such as `Time_Order_picked`), the system:
1. **Predicts delivery ETA with high precision:** Reduces Mean Absolute Error (MAE) from **7.50 minutes (Baseline)** down to **3.12 minutes (Gradient Boosting)**, achieving an **$R^2$ of 0.8210**.
2. **Computes calibrated 80% ETA uncertainty intervals:** An ETA of 24 min is presented as `[19.3, 29.1] min`, with an observed empirical coverage of exactly **80.0%** on held-out test data.
3. **Identifies high-risk late deliveries at order placement:** Achieves a test **ROC-AUC of 0.9853**, **PR-AUC of 0.9703**, and **F1-score of 0.8844**.
4. **Applies cost-sensitive threshold optimization:** Shifts the decision threshold from arbitrary 0.50 down to **0.28**, saving substantial penalty costs by balancing missed delivery churn against dispatch intervention costs.
5. **Pinpoints dark-store expansion priorities:** Pinpoints Zone 3 (62 stores, 39.1% late rate, +9.15 percentage points above network) as the top dark-store candidate.
6. **Detects 33 surge-staffing operational bottlenecks:** Identifies weekday dinner windows (20:00–21:00 on Wed/Fri/Mon) where late rates surge past 55% (+25 pp above network).

---

## 2. Key System Features & Core Innovations

### Key System Features
- **Delivery ETA Prediction Using Machine Learning:** Continuous regression predicting delivery duration with a holdout test MAE of **3.12 minutes** and an $R^2$ of **0.8210**, accompanied by verified 80% coverage prediction intervals ($[-4.70, +5.10]\text{ min}$).
- **Late-Delivery Probability Estimation:** Calibrated class probability outputs evaluating the likelihood of breaching the 30-minute SLA benchmark.
- **Order-Level Risk Classification:** Real-time risk categorization achieving **0.9853 ROC-AUC** and **0.8844 F1**, tuned to an asymmetric cost-optimal threshold of **0.28** (5:1 penalty model).
- **Geographical Demand-Zone Clustering:** Unsupervised spatial K-Means ($k=4$) isolating under-served geographic delay epicenters.
- **Weather, Traffic, and Temporal Delivery Analysis:** Deep multidimensional breakdown of dinner surges (19:00–22:00, 55.4% late rate), traffic friction multipliers (+14 min), and adverse atmospheric conditions.
- **Model Explainability Through Permutation Feature Importance:** Interpretable diagnostic ranking identifying `Road_traffic_density`, `multiple_deliveries`, `distance_km`, and `Weatherconditions` as the top causal drivers.
- **Operational Risk Summaries & Recommendations:** Actionable ROI-driven strategies targeting real-time priority dispatching, micro-fulfillment dark stores, and dynamic surge rostering.
- **Interactive Streamlit Dashboard Prototype:** Production-grade 6-tab operational interface featuring real-time triage, what-if planning simulators, model benchmarks, and cluster maps.

### Five Strategic Innovations
1. **Dual-Model Intelligence:** Combines continuous ETA regression with cost-calibrated late risk classification in a single joint-inference pass at checkout.
2. **Risk-Aware Order Prioritization:** Triages incoming orders dynamically into Priority 1 (Urgent/At-Risk couriers), Priority 2 (Standard), and Priority 3 (Batchable for pooled routing).
3. **Geographic Demand & Delivery-Performance Intelligence:** Combines spatial K-Means clustering with delay severity metrics to map network bottlenecks and dark-store candidates (Zone 3: 39.12% late rate).
4. **Explainable Machine-Learning Predictions:** SHAP factor attributions detailing the exact marginal delay contribution of traffic, weather, distance, and batching.
5. **Foundation for Rider-Allocation Simulation & Dark-Store Planning:** Interactive what-if planning simulator to model the exact operational ROI of adding fleet surge capacity and opening micro-fulfillment hubs.

---

## 3. Data Cleaning & Trust Audit

From `results/cleaning_log.csv`:
- **Raw Volume:** 45,593 rows × 20 columns
- **Removed Rows:** 4,096 rows (9.0% total attrition)
  - 3,640 records with `(0.0, 0.0)` merchant/customer coordinate sentinels
  - 431 records with erroneous Haversine distance > 100 km
  - 25 records with impossible courier age (<16 or >65)
- **String Cleaning & Whitespace Normalization:** Applied to all 15 object columns.
- **Prefix Cleaning:** Removed `"conditions "` from weather and `"(min) "` from delivery duration.
- **Missing Value Conversion & Imputation:**
  - `Delivery_person_Age`: 1,854 string `'NaN'` converted → imputed with median (30.0 yrs)
  - `Delivery_person_Ratings`: 1,908 string `'NaN'` converted → imputed with median (4.70)
  - `Time_Orderd`: 1,731 string `'NaN'` converted → extracted hour/minute imputed with median (19:30)
  - `Weatherconditions`: 616 string `'NaN'` converted → mode imputed (`Fog`)
  - `Road_traffic_density`: 601 string `'NaN'` converted → mode imputed (`Low`)
  - `multiple_deliveries`: 993 string `'NaN'` converted → median imputed (`1.0`)
  - `Festival`: 228 string `'NaN'` converted → mode imputed (`No`)
  - `City`: 1,200 string `'NaN'` converted → mode imputed (`Metropolitian`)
- **Leakage Elimination:** `Time_Order_picked` and `Order_Date` dropped from modeling pipelines.

---

## 3. Modeling Performance Benchmarks

### 3.1. Regression Performance (Delivery Time Prediction)
*Evaluated on independent 8,300-order holdout test set.*

| Model | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Validation MAE (min) | Train MAE (min) |
|---|---|---|---|---|---|
| **Baseline (Mean Predictor)** | 7.5010 | 9.2792 | -0.0001 | 7.6455 | 7.6071 |
| **Ridge Regression (L2)** | 4.7712 | 5.9734 | 0.5855 | 4.7601 | 4.7994 |
| **Random Forest (200 trees)** | 3.1221 | 3.9421 | 0.8195 | 3.1695 | 2.4156 |
| **Gradient Boosting (Best)** | **3.1214** | **3.9261** | **0.8210** | **3.1456** | **3.0105** |

**Uncertainty Quantification:**
- Target 80% Prediction Interval: Residual Quantiles $[Q_{0.10}, Q_{0.90}] = [-4.70, +5.10]\text{ min}$
- Observed Holdout Coverage: **80.0%**

---

### 3.2. Classification Performance (Late Delivery Risk: SLA > 30 min)
*Network baseline late rate = 29.98%. Evaluated on 8,300 holdout test orders.*

| Classifier | Test ROC-AUC | Test PR-AUC | Test F1-Score | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **Baseline (Stratified Dummy)** | 0.5004 | 0.2927 | 0.2913 | 0.2930 | 0.2895 |
| **Logistic Regression (Balanced)** | 0.9373 | 0.8871 | 0.7739 | 0.7162 | 0.8418 |
| **Random Forest (Balanced)** | 0.9833 | 0.9665 | 0.8770 | 0.8659 | 0.8884 |
| **Gradient Boosting (Best)** | **0.9853** | **0.9703** | **0.8844** | **0.8640** | **0.9057** |

---

## 4. Cost-Sensitive Threshold Optimization

A standard 0.50 classification threshold assumes symmetric costs for false positives and false negatives. In quick-commerce delivery operations:
- **Cost of False Negative (Missed Late Order):** Results in customer disappointment, refund claims, support tickets, and churn ($C_{FN} = 5$).
- **Cost of False Positive (Unnecessary Priority Intervention):** Incurs small proactive communication or priority dispatch cost ($C_{FP} = 1$).

### Sensitivity Schedule:
| Cost Ratio ($C_{FN} / C_{FP}$) | Optimal Decision Threshold | Expected Loss on Validation Set |
|---|---|---|
| 2:1 (Conservative penalty) | **0.51** | 674.0 units |
| **5:1 (Operational baseline)** | **0.28** | **908.0 units** |
| 10:1 (High churn penalty) | **0.16** | 1,085.0 units |

At the **0.28 threshold**, recall increases to over **94%**, virtually eradicating surprise delivery delays for high-value customers while controlling operational overhead.

---

## 5. Geographic Clustering: Dark-Store Expansion Analysis

Using K-Means spatial and operational clustering ($k=4$, silhouette score = 0.52):

| Zone ID | Merchant Count | Order Volume | Mean Delivery (min) | Late Rate | Distance (km) | Excess Late (pp) | Priority Index |
|---|---|---|---|---|---|---|---|
| **Zone 3** | **62** | **4,469** | **27.92** | **39.12%** | **10.11** | **+9.15 pp** | **5.07** |
| Zone 0 | 35 | 3,043 | 26.29 | 29.23% | 9.41 | -0.75 pp | 0.02 |
| Zone 1 | 219 | 29,311 | 26.11 | 28.57% | 9.59 | -1.41 pp | 0.02 |
| Zone 2 | 72 | 4,674 | 26.00 | 28.80% | 10.29 | -1.17 pp | 0.02 |

**Strategic Directive:** Zone 3 exhibits a **39.12% late rate**, concentrated around coordinates `(22.548°N, 77.428°E)`. Deploying a dedicated micro-fulfillment dark store at this node directly attacks the network's most severe operational bottleneck.

---

## 6. Courier Behavioral Segmentation

Aggregating performance across riders with $\ge 10$ historical trips:
- **Segment 1 — High-Volume Core Couriers (615 couriers):** 24.66% late rate, 25.25 min avg duration, 4.66 customer rating, 38.4 orders handled on average.
- **Segment 0 — Long-Distance Specialists (481 couriers):** 37.60% late rate, 27.81 min avg duration, 4.61 customer rating, 35.9 orders handled on average.

**Intervention:** Couriers in Segment 0 should be paired with short-radius batch routes or provided vehicle battery/maintenance subsidies to mitigate late-stage trip fatigue.

---

## 7. Surge-Staffing Rule & Operational Heatmap

Analysis of `results/surge_windows.csv` revealed **33 distinct hour × day-of-week surge windows** exceeding network late rates by $>5$ percentage points:
- **Top 1:** Wednesday 20:00–21:00 (801 orders, **55.43% late rate**, +25.46 pp excess)
- **Top 2:** Friday 21:00–22:00 (677 orders, **54.36% late rate**, +24.38 pp excess)
- **Top 3:** Friday 20:00–21:00 (654 orders, **53.52% late rate**, +23.54 pp excess)
- **Top 4:** Wednesday 21:00–22:00 (764 orders, **53.01% late rate**, +23.03 pp excess)
- **Top 5:** Sunday 21:00–22:00 (599 orders, **51.75% late rate**, +21.78 pp excess)

**Execution Rule:** Dynamically schedule **25–30% extra rider capacity** on Wednesday, Friday, and weekend dinner windows (19:30–21:30) to suppress structural congestion delays.
