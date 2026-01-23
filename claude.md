# Project Transfer Merton
**Scope:** Tractable Time-Series Analysis of Civic System Latency in SW19.
**Objective:** Model the transfer function between economic "shocks" and food bank demand.

---

## 1. System Architecture & Intent
This project treats the Merton social safety net as a dynamic system. We aim to identify the systemic response time ($t_{lag}$) following a delta-function economic stimulus (e.g., energy price spike).

### Core Hypothesis
The demand $D(t)$ at the Wimbledon Foodbank is an impulse response of the "vulnerability field" $V(x, t)$ to exogenous shocks $S(t)$:
$$D(t) = (S * h)(t) + \eta(t)$$
Where $h(\tau)$ is the **Systemic Vulnerability Kernel** we seek to characterise.

---

## 2. Technical Stack
- **Language:** Python 3.10+
- **Data Orchestration:** `polars` (preferred over pandas for memory efficiency/speed)
- **Signal Processing:** `scipy.signal` for cross-correlation and convolution.
- **Modeling:** `Prophet` for baseline seasonality; `PyMC` for Bayesian inference of the kernel shape.
- **Hardware Target:** Raspberry Pi 5 (for the dashboard/automated ingestion).

---

## 3. Data Sources & Schema
### A. Ingestion Points
1. **Nomis API (NM_162_1):** Monthly Claimant Count for Merton (Geo: 1946157274).
2. **Octopus Agile API:** Hourly/Daily energy price volatility index.
3. **Merton Open Data:** Ward-level IMD (Index of Multiple Deprivation) for SW19 spatial weighting.

### B. Feature Engineering
- **Energy-Degree Days:** $\max(0, T_{base} - T_{avg})$ weighted by local EPC density.
- **Friction Index:** Derivative of claimant count $dC/dt$ to detect policy "shocks."

---

## 4. Coding Conventions
- **Explain & Document:** All functions must include a docstring detailing the physics or statistical rationale (e.g., "Applying a Savitzky-Golay filter to preserve transients in high-frequency energy data").
- **Strict Typing:** Use Python type hints for all data transformation pipelines.
- **No Hype:** Use "Predictive Latency" instead of "AI Forecasting."

---

## 5. Development Workflow
- `/data/raw`: Immutable source files.
- `/notebooks`: Exploratory Signal Analysis (ESA).
- `/src/models`: Implementation of the Vulnerability Kernel.
- `/src/api`: Nomis and Octopus connectors.

---

## 6. Policy Influence Strategy
**Primary Output:** A "Systemic Latency Report" for SW19 ward councillors. 
**Metric for Success:** Identification of a stable $\tau_{lag}$ that allows for intervention $t < \tau_{lag}$ to prevent acute crisis.