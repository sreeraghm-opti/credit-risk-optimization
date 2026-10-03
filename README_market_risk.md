# Market Risk Analytics: VaR, Expected Shortfall and Backtesting

## Overview

This project implements a quantitative market-risk framework using historical NIFTY 50 index data. It estimates Value at Risk (VaR) and Expected Shortfall (ES), evaluates one-day-ahead forecasts, and compares historical, parametric normal, EWMA normal, and Student-t approaches.

## Data

* **Instrument:** NIFTY 50 index (`^NSEI`)
* **Source:** Yahoo Finance via `yfinance`
* **Historical period:** 2 January 2015 – 1 October 2026
* **Price observations:** 2,893
* **Common backtesting observations:** 2,642

The index-price series is not an investable total-return series. Results exclude dividends, transaction costs, and portfolio-specific effects. The illustrative ₹10 million exposure is a scaling assumption, not an actual portfolio.

## Models

1. Historical VaR and Expected Shortfall
2. Parametric normal VaR and Expected Shortfall
3. Monte Carlo normal VaR and Expected Shortfall
4. EWMA volatility with normal innovations
5. EWMA volatility with Student-t innovations
6. Corrected EWMA Student-t model using volatility-standardised historical residuals

VaR is evaluated at 95% and 99% confidence levels.

## Backtesting methodology

The framework performs rolling one-day-ahead forecasts using a 250-observation historical window. A VaR exception occurs when the realised loss exceeds the forecast VaR.

Statistical tests include:

* Kupiec unconditional coverage test
* Christoffersen first-order independence test
* Conditional coverage test combining coverage and independence

All five models are compared on common forecast dates.

## Key findings

### 95% VaR

The EWMA normal model was not rejected by the conditional coverage test at the 5% significance level (p = 0.2017). The corrected EWMA Student-t model was also not rejected (p = 0.0962).

The historical and parametric normal models were rejected by the conditional coverage test, with evidence of clustered exceptions during the evaluation period.

### 99% VaR

All five approaches were rejected by the conditional coverage test at the 5% significance level. The corrected EWMA Student-t model recorded 41 exceptions, compared with approximately 26.42 expected under correct 99% coverage.

None of the evaluated approaches adequately calibrated 99% VaR over this backtesting period.

Failure to reject a model does not prove it is correct. Results depend on the sample, model assumptions, and testing method.

## Principal outputs

Files are saved in `reports/market/`.

| File                                   | Description                                  |
| -------------------------------------- | -------------------------------------------- |
| `var_model_comparison_summary.csv`     | Model performance and statistical tests      |
| `var_model_comparison_daily.csv`       | Daily forecasts and realised outcomes        |
| `var_model_comparison_report.md`       | Written comparison report                    |
| `var_model_comparison_95.png`          | Visual comparison at 95% confidence          |
| `var_model_comparison_99.png`          | Visual comparison at 99% confidence          |
| `var_backtest_summary.csv`             | Historical and normal-model backtest results |
| `var_backtest_diagnostics.csv`         | Coverage and independence diagnostics        |
| `ewma_backtest_summary.csv`            | EWMA normal backtest results                 |
| `ewma_student_t_backtest_summary.csv`  | Original Student-t results                   |
| `ewma_student_t_corrected_summary.csv` | Corrected Student-t results                  |

## Reproducing the analysis

Run from the project root:

```bash
.venv/bin/python src/market_risk/download_market_data.py
.venv/bin/python src/market_risk/var_engine.py
.venv/bin/python src/market_risk/backtest_var.py
.venv/bin/python src/market_risk/backtest_diagnostics.py
.venv/bin/python src/market_risk/ewma_backtest.py
.venv/bin/python src/market_risk/ewma_student_t_backtest.py
.venv/bin/python src/market_risk/ewma_student_t_corrected.py
.venv/bin/python src/market_risk/compare_var_models.py
```

Downloading market data requires an internet connection. Results may change when the analysis is rerun with updated data.

## Limitations

* Historical returns may not represent future market regimes.
* Normal innovations can underrepresent extreme-tail risk.
* Statistical tests have limited power, particularly with relatively few 99% VaR exceptions.
* Results are sensitive to window length and model specification.
* This is an independent research project, not a production risk system or regulatory capital calculation.
* Results are illustrative and are not investment advice.

## Skills demonstrated

Python, pandas, NumPy, SciPy, time-series analysis, volatility modelling, Monte Carlo simulation, Value at Risk, Expected Shortfall, hypothesis testing, model validation, data visualisation, and reproducible quantitative research.
