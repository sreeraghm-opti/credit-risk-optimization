# Credit & Market Risk Analytics

A reproducible quantitative risk analytics project combining **credit-risk modelling, portfolio risk, market-risk measurement, volatility modelling, Monte Carlo simulation, and statistical backtesting** using Python.

The project is designed as an applied quantitative finance research portfolio, with emphasis on model validation, uncertainty, stress testing, and reproducibility.

---

## Project Overview

This project develops two complementary risk analytics frameworks:

### 1. Credit Risk

A credit-default modelling and portfolio-risk framework using the UCI Default of Credit Card Clients dataset.

The analysis covers:

- Exploratory data analysis
- Logistic regression
- Random forest modelling
- Probability prediction
- Model comparison
- Probability calibration
- Cross-validation
- Fairness diagnostics
- Expected Credit Loss (ECL) estimation
- Monte Carlo portfolio loss simulation
- Stress testing
- Threshold analysis
- Constrained portfolio optimisation
- Portfolio robustness and sensitivity analysis

### 2. Market Risk

A market-risk engine based on historical NIFTY 50 price data.

The framework implements:

- Historical VaR
- Parametric normal VaR
- Monte Carlo VaR
- Expected Shortfall
- EWMA volatility modelling
- EWMA Student-t volatility modelling
- Rolling one-day-ahead VaR forecasts
- Kupiec unconditional coverage testing
- Christoffersen independence testing
- Conditional coverage testing
- Comparative VaR model diagnostics

---

## Key Results

### Credit Risk

Using out-of-fold predictions:

| Model | ROC-AUC | Average Precision | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.724 | 0.504 | 0.145 |
| Random Forest | 0.780 | 0.555 | 0.135 |

The analysis also evaluates probability calibration and group-level model behaviour rather than relying only on discrimination metrics.

A constrained portfolio-selection framework was developed using:

- Exposure constraints
- Expected-loss budgets
- Maximum account limits
- Assumed Loss Given Default
- Credit-limit-based EAD proxy

These results are illustrative research estimates rather than production lending decisions or regulatory capital calculations.

### Market Risk

The market-risk framework evaluates **2,642 rolling one-day-ahead forecasts** across 95% and 99% confidence levels.

The analysis compares:

- Historical VaR
- Parametric normal VaR
- EWMA normal VaR
- EWMA Student-t VaR
- Corrected EWMA Student-t VaR

At the 95% level, several models are not rejected by conditional-coverage testing in this sample, while the 99% level provides a substantially more demanding tail-risk test.

The analysis demonstrates why a model should not be judged solely by the number of VaR exceptions: exception frequency, clustering, and joint coverage behaviour all matter.

---

## Market Risk Methodology

The market-risk engine follows a rolling forecasting framework.

For each forecast date:

1. A trailing historical window is constructed.
2. The risk model is estimated using information available up to that date.
3. One-day-ahead VaR is forecast.
4. The realised return is compared with the forecast.
5. Exceptions are recorded.
6. Statistical backtests are applied.

### VaR Backtesting

The project implements:

**Kupiec Unconditional Coverage Test**

Tests whether the observed exception frequency is consistent with the nominal VaR confidence level.

**Christoffersen Independence Test**

Tests whether VaR exceptions exhibit first-order clustering.

**Christoffersen Conditional Coverage Test**

Combines unconditional coverage and independence into a joint test.

This allows the analysis to distinguish between:

- Too many exceptions
- Correct exception frequency but clustered exceptions
- Failure of both coverage and independence

---

## Volatility Modelling

The project implements EWMA volatility forecasting and Student-t extensions.

The Student-t implementation was subsequently corrected to estimate the distribution of **EWMA-standardised innovations**, rather than fitting the Student-t distribution directly to raw returns using a single unconditional standard deviation.

This provides a more internally consistent treatment of conditional volatility and heavy-tailed innovations.

---

## Credit Risk Methodology

The credit-risk framework combines predictive modelling with portfolio-level risk analysis.

### Probability of Default

Models generate estimated probabilities of default rather than relying exclusively on binary classifications.

### Expected Credit Loss

The project uses:

\[
ECL = PD \times LGD \times EAD
\]

where:

- **PD** = probability of default
- **LGD** = assumed loss given default
- **EAD** = exposure at default proxy

Credit limit is used as an exposure proxy in this research implementation.

### Monte Carlo Portfolio Risk

Portfolio losses are simulated under an assumed correlation structure to examine:

- Expected portfolio loss
- Loss dispersion
- Value at Risk
- Expected Shortfall

These assumptions are explicitly treated as illustrative rather than regulatory estimates.

### Stress Testing

The framework evaluates alternative PD scenarios to examine how portfolio expected loss changes under adverse assumptions.

---

## Portfolio Optimisation

A constrained optimisation framework evaluates portfolio composition subject to:

- Maximum exposure
- Expected-loss budget
- Maximum number of accounts
- Assumed LGD
- Exposure-at-default proxy

Sensitivity analysis evaluates how portfolio composition changes under alternative:

- LGD assumptions
- Expected-loss budgets
- Exposure limits

Portfolio overlap is also examined using Jaccard similarity to assess selection stability.

---

## Data

### Credit Risk Dataset

**Default of Credit Card Clients Dataset**

Source: UCI Machine Learning Repository.

The dataset contains 30,000 observations relating to credit-card default behaviour.

The analysis is based on historical data and should not be interpreted as a current representation of any particular bank's customer population.

### Market Data

Historical NIFTY 50 price data downloaded using `yfinance`.

The market dataset covers:

**2015-01-02 to 2026-10-01**

The analysis uses index prices rather than an investable total-return series and therefore does not incorporate dividends, transaction costs, or implementation effects.

---

## Repository Structure

```text
credit-risk-optimization/
│
├── data/
│   └── raw/
│       ├── credit_card_default.csv
│       └── market/
│           └── nifty50_prices.csv
│
├── reports/
│   ├── Credit-risk model results
│   ├── Portfolio optimisation results
│   ├── Stress-testing results
│   └── Market-risk backtesting results
│
├── src/
│   └── market_risk/
│       ├── var_engine.py
│       ├── backtest_var.py
│       ├── backtest_diagnostics.py
│       ├── ewma_backtest.py
│       ├── ewma_student_t_backtest.py
│       ├── ewma_student_t_corrected.py
│       ├── compare_var_models.py
│       └── download_market_data.py
│
├── tests/
│   └── test_market_risk.py
│
├── app.py
├── baseline_model.py
├── model_comparison.py
├── model_validation.py
├── expected_credit_loss.py
├── monte_carlo_risk.py
├── stress_testing.py
├── portfolio_optimization.py
├── portfolio_robustness.py
├── portfolio_robustness_corrected.py
├── fairness_audit.py
├── threshold_analysis.py
├── eda.py
└── README.md