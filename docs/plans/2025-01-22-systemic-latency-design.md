# Systemic Latency Analysis Design

**Date:** 2025-01-22
**Status:** Approved
**Goal:** Model the transfer function between economic stress indicators and food insecurity to identify intervention windows for SW19 policy.

---

## 1. Overall Approach

### Core Model

Economic Fairness indicators act as *leading signals* for food insecurity. The transfer function framework:

```
Food_Insecurity(t) = Σ [β_i × EconomicIndicator_i(t - τ_i)] + seasonality + noise
```

### Three-Phase Analysis

1. **Correlation study** — For each Economic Fairness indicator, compute cross-correlation with Food Foundation time series at various lags. Identify which indicators show statistically significant leading relationships.

2. **Lag quantification** — For the strongest correlates, estimate τ_lag using `scipy.signal.correlate` and Bayesian inference (PyMC) to get uncertainty bounds on the delay.

3. **Kernel estimation** — Fit the "Systemic Vulnerability Kernel" h(τ) — the shape of how a shock propagates through time.

### Scope Limitation

Both datasets are relatively sparse: Economic Fairness has ~6 years of annual/quarterly data; Food Foundation has 17 survey rounds over 4 years. Statistical power will be limited — findings framed as "indicative patterns requiring validation" rather than definitive causal claims.

### Geographic Approach

Demonstrate methodology at national level. Frame report as "applicable to Merton pending local data availability."

---

## 2. Data Pipeline & Ingestion

### Data Source A: Economic Fairness Indicators

From the [London Datastore](https://data.london.gov.uk/economic-fairness/), prioritised indicators:

| Indicator | Rationale | Expected Lag Role |
|-----------|-----------|-------------------|
| Absolute Poverty rate | Direct vulnerability measure | Coincident/lagging |
| Disposable Income | Spending capacity | Leading |
| Employment Gaps | Job loss precedes hardship | Leading |
| London Living Wage coverage | Income adequacy | Leading |
| Income Inequality (Gini) | Structural vulnerability | Slow-moving baseline |

### Data Source B: Food Insecurity Tracker

From [Food Foundation](https://foodfoundation.org.uk/initiatives/food-insecurity-tracking) 17 survey rounds:
- % of households experiencing food insecurity
- % of households with children affected
- Survey dates (to align time axis)

### Directory Structure

```
/data/raw/
  economic_fairness/
    indicator_matrix_nov_2020.xlsx
    poverty_timeseries.csv
    income_timeseries.csv
  food_insecurity/
    survey_rounds.csv
```

### Ingestion Strategy

1. Download Economic Fairness Excel; parse with `polars`
2. Manually extract Food Foundation survey results from published rounds into CSV (17 rows, ~5 columns)

### Time Alignment

Resample both to monthly timestamps. Interpolate Economic Fairness (sparse) with cubic spline; treat Food Foundation survey dates as point observations.

---

## 3. Analysis Methodology

### Phase 1: Correlation Study

For each Economic Fairness indicator × Food Insecurity pairing:

```python
# Cross-correlation at lags from -12 to +12 months
lags, correlation = scipy.signal.correlation_lags(...)
```

Output: Heatmap showing which indicators correlate with food insecurity and at what lag. Indicators where peak correlation occurs at negative lag (indicator leads) are candidate predictors.

### Phase 2: Lag Quantification

For significant correlates, estimate τ_lag with uncertainty using PyMC:

```python
with pm.Model():
    tau = pm.Normal("tau_lag", mu=3, sigma=2)  # prior: ~3 months
    beta = pm.Normal("beta", mu=0, sigma=1)

    predicted = beta * shift(indicator, tau)
    likelihood = pm.Normal("obs", mu=predicted, sigma=sigma, observed=food_insecurity)
```

Output: Posterior distribution over τ_lag (e.g., "2.4 months, 95% CI: 1.1–4.2").

### Phase 3: Kernel Shape

Test three candidate kernel shapes for h(τ):

| Shape | Interpretation |
|-------|----------------|
| Delta | Instantaneous pass-through |
| Exponential decay | Shock hits immediately, effect fades |
| Gamma distribution | Delayed peak, then decay |

Use Bayesian model comparison (WAIC/LOO) to select best-fitting kernel.

### Analysis Outputs

- Ranked list of leading indicators with lag estimates
- Visualisation of the fitted kernel h(τ)
- "Intervention window" estimate

---

## 4. Notebook Structure

```
/notebooks/
  01_data_ingestion.ipynb       ← Load, validate, align time series
  02_exploratory_viz.ipynb      ← Plot raw series, check seasonality
  03_correlation_study.ipynb    ← Cross-correlation heatmaps
  04_lag_estimation.ipynb       ← PyMC Bayesian inference
  05_kernel_fitting.ipynb       ← Compare kernel shapes
  06_synthesis.ipynb            ← Key findings, figures for report
```

### Notebook Details

**01_data_ingestion.ipynb**
- Parse Economic Fairness Excel with `polars`
- Load manually-compiled Food Foundation CSV
- Resample to common monthly index
- Output: `/data/processed/aligned_timeseries.parquet`

**02_exploratory_viz.ipynb**
- Plot each indicator over time
- Overlay Food Foundation survey points
- Check for trends, seasonality, outliers
- Apply Savitzky-Golay filter to smooth noise while preserving transients

**03-05: Core Analysis**
- Each notebook produces artifacts (figures, posterior samples) saved to `/outputs/`

**06_synthesis.ipynb**
- Compile key findings
- Generate publication-ready figures
- Draft report narrative

---

## 5. Policy Report Deliverable

### Target Document: "Systemic Latency Report for SW19"

```
1. Executive Summary (1 page)
   - Key finding: τ_lag estimate in plain language
   - "Economic stress signals precede food insecurity by X months"
   - Intervention window implication

2. Background (0.5 page)
   - What is systemic latency and why it matters
   - Data sources used

3. Findings (2 pages)
   - Which indicators are leading signals (ranked)
   - Visualisation: "Early Warning Timeline"
   - Confidence levels and limitations

4. Policy Implications (1 page)
   - Interventions that fit the τ_lag window
   - Monitoring recommendations

5. Methodology Note (appendix)
   - Technical details for scrutiny
   - Link to notebooks for reproducibility

6. Next Steps
   - "Pending Merton-specific data, methodology can be localised"
   - Suggested local data sources to pursue
```

### Key Visual: Intervention Window Diagram

```
Economic     ──────┐
Shock              │
                   ▼
              ┌─────────┐
              │  τ_lag  │  ← Intervention Window
              └─────────┘
                   │
                   ▼
Food         ──────┘
Insecurity
Crisis
```

---

## 6. Technical Stack

As defined in project CLAUDE.md:
- **Language:** Python 3.10+
- **Data:** `polars`
- **Signal Processing:** `scipy.signal`
- **Modeling:** `Prophet` (seasonality), `PyMC` (Bayesian inference)
- **Visualisation:** `matplotlib`, `arviz`

---

## 7. Next Steps

1. Download and stage raw data files
2. Manually compile Food Foundation survey data into CSV
3. Begin with `01_data_ingestion.ipynb`
4. Progress through notebook sequence
5. Synthesise findings into policy report
