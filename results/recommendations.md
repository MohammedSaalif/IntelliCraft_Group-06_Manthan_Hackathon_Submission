# Business Recommendations & Strategic Innovations

## Context
- SLA: 30 minutes (candidate)
- Best regression model: test MAE = 3.12 min, $R^2 = 0.8210$, ETA band coverage = 80.0%
- Best classifier: test ROC-AUC = 0.9853, PR-AUC = 0.9703, F1 = 0.8844, cost-optimal threshold = 0.28
- Surge windows: 33 hour×weekday slots with excess late rate (dinner peaks up to 55.4%)

---

## Strategic Innovations Powering Recommendations
1. **Dual-Model Intelligence:** Real-time synchronous inference combining continuous ETA regression (3.12 min MAE) with cost-calibrated late risk classification (0.9853 ROC-AUC).
2. **Risk-Aware Order Prioritization:** Automated 3-tier dispatch queuing (P1: Urgent/At-Risk, P2: Standard, P3: Batchable).
3. **Geographic Demand & Delivery-Performance Intelligence:** Spatial clustering mapping under-served delay zones (Zone 3: 39.12% late rate).
4. **Explainable Machine-Learning Predictions:** Feature attribution quantifying marginal delay impacts.
5. **Foundation for Rider-Allocation Simulation & Dark-Store Planning:** Interactive simulation tools for fleet surge and expansion planning.

---

## Recommendation 1: Deploy Real-Time Late-Order Flagging System
**Impact: HIGH | Effort: MEDIUM**

Deploy the trained classifier at order placement to flag orders with >28% probability of being late. For flagged orders:
- Assign priority riders (experienced, high-rated)
- Send proactive customer communication with dynamic ETA and uncertainty bounds
- Trigger dynamic pricing incentives for nearby riders

**Expected benefit:** Using the cost-optimal threshold (0.28) minimizes misclassification costs compared to the default 0.50 threshold by balancing missed late penalties against unnecessary interventions.

**Assumption:** Cost of a missed late order is 5× the cost of unnecessary intervention. Actual costs should be calibrated with business data.

---

## Recommendation 2: Open Dark Stores in High-Delay Zones
**Impact: HIGH | Effort: HIGH**

Zone analysis identified 4 geographic clusters. The highest-priority zone (Zone 3):
- Late rate: 39.1% vs network average
- Excess late rate: +9.1 percentage points
- Center location: (22.5479, 77.4284)

**Action:** Open a dark store / micro-fulfillment center near the top-priority zone center. This reduces last-mile distance, cutting delivery time.

**Expected benefit:** Reducing average distance in the top zone by 30% could lower delivery times by ~1 minutes (ESTIMATE based on distance-time relationship observed in the data).

---

## Recommendation 3: Targeted Rider Training & Dynamic Surge Staffing
**Impact: MEDIUM | Effort: LOW**

**Rider Training:** Segment analysis found 2 rider segments. The "Long-Distance Specialist" segment (481 riders) has a late rate of 37.6%. Provide targeted training (route optimization, time management) to this group.

**Surge Staffing:** 33 hour×weekday windows show excess late rates. Schedule additional riders during these windows:
|   order_hour |   day_of_week |   n_orders |   late_rate |   excess_pp |
|-------------:|--------------:|-----------:|------------:|------------:|
|           20 |             2 |        801 |    0.554307 |     25.4551 |
|           21 |             4 |        677 |    0.543575 |     24.3818 |
|           20 |             4 |        654 |    0.535168 |     23.5412 |
|           21 |             2 |        764 |    0.530105 |     23.0348 |
|           21 |             6 |        599 |    0.517529 |     21.7773 |

**Expected benefit:** Improving the worst rider segment's late rate to the network average would reduce overall late deliveries by an estimated amount based on the segment's share of total riders.

---

## Limitations
1. All projections are ESTIMATES based on observed correlations, not causal experiments.
2. The cost ratio (5:1) is an ASSUMPTION.
3. The SLA of 30 min is a candidate; replace with the official organiser SLA if different.
4. Geographic zone analysis is limited to Haversine distance; real-world routing may differ.
5. Rider segmentation uses historical averages; individual rider performance varies over time.
