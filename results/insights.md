# EDA Insights

## Dataset Summary
- **Rows (clean):** 41,497
- **Target:** Time_taken(min), range [10, 54], mean=26.3, median=26.0
- **SLA threshold:** 30 min (candidate)
- **Late rate:** 30.0% (12,439 / 41,497)

## Key Findings
1. **Distance is the strongest continuous predictor** of delivery time (corr ~ 0.32).
2. **City type matters:** Metropolitan cities tend to have different patterns than Urban/Semi-Urban.
3. **Traffic density** is a major factor — Jam conditions significantly increase delivery times.
4. **Festival periods** increase delivery times and late rates.
5. **Peak hours** (lunch & dinner) show elevated late rates.
6. **Rider ratings** have a moderate negative correlation with delivery time.
7. **Multiple deliveries** increase delivery time.
8. **Weather conditions** — Stormy/Sandstorm conditions cause delays.

## Hour × Weekday Late Rate
See `results/hour_weekday_late_rate.csv` for the full table.
Peak late-rate windows: typically dinner hours (19–22) on weekdays.

## Figures
- `results/figures/01_target_distribution.png`
- `results/figures/02_hourly_patterns.png`
- `results/figures/03_correlation_heatmap.png`
- `results/figures/04_categorical_boxplots.png`
- `results/figures/05_distance_vs_time.png`
- `results/figures/06_day_of_week.png`
