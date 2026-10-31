# Team Log — Quick-Commerce Delivery Intelligence
## KLU + PATH3 ML-01 | Manthan Hackathon

---

## Team Members & Roles

| Member | Role | Core Responsibilities |
|---|---|---|
| **Mohammed Saalif** | Data & Insights Lead | Data cleaning, feature engineering, EDA |
| **[P2 Name]** | Regression & App Lead | Delivery time prediction models, Streamlit app |
| **[P3 Name]** | Classification & Risk Lead | Late-delivery classifier, cost threshold tuning |
| **[P4 Name]** | Clustering & Business Lead | Spatial clustering, business recommendations, reports |

---

## Contributions Summary

### Mohammed Saalif — Data & Insights Lead
- Built the end-to-end data cleaning pipeline (`src/cleaning.py`), removing 4,096 invalid records (GPS sentinels, distance outliers, age bounds, literal NaN strings).
- Engineered the zero-leakage feature set (`src/features.py`) with Haversine distance, temporal peak flags, and a strict elimination of `Time_Order_picked`.
- Conducted exploratory data analysis producing 6 diagnostic figures covering target distribution, hourly patterns, traffic density, weather impact, distance correlation, and day-of-week effects.
- Authored the data dictionary (`reports/data_dictionary.md`) documenting all 20 columns, transformations, and imputation decisions.

### [P2 Name] — Regression & App Lead
- Designed and benchmarked four regression models (Baseline, Ridge, Random Forest, Gradient Boosting) in `src/regression.py`.
- Achieved a test MAE of **3.12 minutes** and $R^2$ of **0.8210** with Gradient Boosting on 8,300 holdout orders.
- Derived empirical 80% coverage prediction intervals $[-4.70, +5.10]$ min via validation residual quantiles.
- Built the interactive 6-tab Streamlit dashboard (`app/streamlit_app.py`) with real-time order triage, model benchmarks, and planning simulators.

### [P3 Name] — Classification & Risk Lead
- Benchmarked four classifiers (Baseline, Logistic Regression, Random Forest, Gradient Boosting) in `src/classification.py` for 30-min SLA risk prediction.
- Achieved **0.9853 ROC-AUC** and **0.8844 F1** on the holdout test set.
- Implemented asymmetric cost-sensitive threshold optimization, shifting the decision boundary from 0.50 to **0.28** under a 5:1 penalty model to minimize business loss.
- Identified **33 surge-staffing bottleneck windows** with late rates exceeding 50% (e.g. Wednesday 20:00–21:00 at 55.4%).

### [P4 Name] — Clustering & Business Lead
- Built the spatial K-Means clustering pipeline (`src/clustering.py`) identifying 4 delivery zones and isolating **Zone 3** (39.1% late rate, +9.15 pp excess) as the top dark-store candidate.
- Profiled 2 courier behavioral segments to guide targeted training and route assignment.
- Authored the executive summary (`reports/executive_summary.md`) and 3 prioritized ROI-driven business recommendations (`results/recommendations.md`).
- Orchestrated the master pipeline script (`run_all.py`) enabling full end-to-end reproducibility with a single command.

---

## Key Decisions Made Together

1. **Zero Leakage Boundary** — `Time_Order_picked` dropped from all pipelines; it is only observable after order acceptance, not at placement.
2. **Coordinate Sentinel Removal** — 3,640 records with `(0.0, 0.0)` GPS coordinates purged rather than imputed to preserve spatial integrity.
3. **Model Selection** — Gradient Boosting selected for both tasks based on holdout test performance and fast inference.
4. **Asymmetric Cost Ratio** — 5:1 ($C_{FN}=5, C_{FP}=1$) adopted to reflect that missing a late delivery costs far more than proactive intervention.
5. **Uncertainty Quantification** — Empirical residual quantiles used instead of Gaussian assumptions to achieve verified 80.0% test coverage.
