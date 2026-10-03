# Credit Risk Model Governance Summary

## 1. Model Purpose

This research model estimates the probability that a credit-card
account will default in the subsequent period using the historical
UCI Default of Credit Card Clients dataset.

The model is intended for research and portfolio demonstration.
It is **not a production lending model, regulatory capital model,
or customer-credit decisioning system**.

---

## 2. Primary Model

Random Forest classifier.

Training/test methodology:

- 80/20 stratified train/test split
- Random state: 42
- Five-fold stratified cross-validation for model validation
- Out-of-fold predictions used for validation diagnostics

---

## 3. Discrimination

Random Forest cross-validation results:

- ROC-AUC: 0.7795
- Gini coefficient: 0.5590
- KS statistic: 0.4271
- Average Precision: 0.5549
- Brier score: 0.1346

Bootstrap 95% confidence intervals are reported for ROC-AUC,
Average Precision and Brier score.

---

## 4. Calibration

Random Forest calibration diagnostics:

- Calibration intercept: -0.0083
- Calibration slope: 1.0192

PD-decile analysis compares predicted default probabilities with
observed default frequencies across ten risk-ranked groups.

---

## 5. Feature Explainability

Three complementary approaches are used:

1. Random Forest native feature importance
2. Permutation importance based on ROC-AUC
3. SHAP mean absolute contribution

The strongest SHAP-ranked feature is:

- pay_status_sep

The highest-IV feature is:

- pay_status_sep
- IV: 0.8790

Agreement across multiple importance methodologies provides
additional evidence about the model's principal predictive drivers,
although feature importance does not establish causality.

---

## 6. Population Stability

Feature-level PSI was calculated between the random training and
test samples using training-derived bins.

Maximum observed PSI:

- 0.0026

This indicates very little distributional difference between the
random training and test samples.

This is **not equivalent to temporal stability testing**.
Production monitoring would require genuine out-of-time samples
and ongoing population monitoring.

---

## 7. Sensitive Features

The dataset contains the `sex` variable, which is included in
the model comparison.

A separate fairness analysis is included in the project.

Model performance differences across groups should not be treated
as evidence of legal compliance or fairness certification.

Any production use would require appropriate legal, compliance,
fair-lending and model-governance review.

---

## 8. Key Model-Risk Limitations

### Historical dataset

The underlying dataset is historical and does not represent a
current bank portfolio.

### Random validation

Random cross-validation does not test performance under future
economic or portfolio conditions.

### No genuine macroeconomic stress model

Stress scenarios in this project are illustrative PD shocks rather
than estimated causal responses to macroeconomic variables.

### Proxy exposure

The portfolio optimization component uses a credit-limit-based
exposure proxy rather than verified contractual EAD.

### Explainability

SHAP and feature importance explain model behaviour but do not
establish causal relationships.

### Production deployment

No production deployment, automated decisioning or customer-facing
adverse-action process is implemented.

---

## 9. Recommended Production Extensions

A production-grade implementation would require:

- Out-of-time validation
- Challenger models
- Macroeconomic stress testing
- Drift monitoring
- Formal model-change controls
- Fair-lending/compliance review
- Independent model validation
- Documentation of data lineage
- Production EAD/LGD estimates
- Recovery and loss-history data
- Monitoring thresholds and escalation procedures
- Customer-facing reason-code governance

---

## 10. Overall Governance Position

This repository demonstrates a research-oriented credit-risk modelling
workflow combining:

- Predictive modelling
- Discrimination analysis
- Probability calibration
- Scorecard diagnostics
- Population stability
- Explainability
- Fairness analysis
- Expected credit loss estimation
- Stress testing
- Portfolio optimization

The results should be interpreted as analytical evidence from a
historical research dataset rather than as evidence of production
model suitability.
