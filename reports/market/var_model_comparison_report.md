# VaR Model Comparison

## Methodology

- Common evaluation dates: 2642
- Models: historical, parametric normal, EWMA normal, original EWMA Student-t, corrected EWMA Student-t.
- Exception: realised loss strictly exceeds the forecast VaR.
- Kupiec tests unconditional coverage; independence tests first-order exception dependence; conditional coverage combines them.
- P-values below 0.05 indicate rejection at the 5% level.

These are historical NIFTY 50 index backtests, not a regulatory validation of a bank trading-book model.

## Results

### 95% confidence

| Model | Exceptions | Expected | Exception rate | Mean VaR | Kupiec p | Independence p | Conditional coverage p | Reject at 5%? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EWMA Student-t (corrected) | 156 | 132.10 | 5.90% | 1.43% | 0.0378 | 0.5427 | 0.0962 | No |
| EWMA Student-t (original) | 194 | 132.10 | 7.34% | 1.26% | 0.0000 | 0.0199 | 0.0000 | Yes |
| EWMA normal | 150 | 132.10 | 5.68% | 1.48% | 0.1175 | 0.3862 | 0.2017 | No |
| Historical VaR | 143 | 132.10 | 5.41% | 1.46% | 0.3367 | 0.0001 | 0.0002 | Yes |
| Parametric normal VaR | 134 | 132.10 | 5.07% | 1.53% | 0.8656 | 0.0000 | 0.0000 | Yes |

### 99% confidence

| Model | Exceptions | Expected | Exception rate | Mean VaR | Kupiec p | Independence p | Conditional coverage p | Reject at 5%? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EWMA Student-t (corrected) | 41 | 26.42 | 1.55% | 2.26% | 0.0084 | 0.6687 | 0.0282 | Yes |
| EWMA Student-t (original) | 49 | 26.42 | 1.85% | 2.13% | 0.0001 | 0.9238 | 0.0004 | Yes |
| EWMA normal | 49 | 26.42 | 1.85% | 2.09% | 0.0001 | 0.9238 | 0.0004 | Yes |
| Historical VaR | 42 | 26.42 | 1.59% | 2.54% | 0.0050 | 0.0316 | 0.0019 | Yes |
| Parametric normal VaR | 49 | 26.42 | 1.85% | 2.19% | 0.0001 | 0.0137 | 0.0000 | Yes |

## Interpretation

A lower exception rate is not automatically better: the objective is calibration to the intended exception probability, alongside adequate independence and risk sensitivity. Failure to reject a test is not proof that a model is correct. The 99% backtest has relatively few expected exceptions, so test power is limited.

Mean VaR is shown as a percentage of index value, not as a portfolio loss estimate. Index returns exclude dividends and do not include transaction costs.
