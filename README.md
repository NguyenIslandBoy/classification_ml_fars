# Traffic Accident Injury Severity Classification

## Overview
Machine learning pipeline for predicting injury severity in US traffic accidents using the FARS (Fatality Analysis Reporting System) dataset.

## Dataset
- **Source**: NHTSA FARS Database
- **Samples**: 99,768 accidents
- **Features**: 115 engineered features (from 20 original)
- **Target**: 4-class injury severity (Fatal, Severe, Minor, No Injury)
- **Class Imbalance**: Fatal (42%), Minor (23%), No Injury (20%), Severe (15%)

## Pipelines Implemented

| Pipeline | Test Macro-F1 | Training Time |
|----------|---------------|---------------|
| Logistic Regression (Baseline) | 0.5203 | 10.4 sec |
| Random Forest | 0.5386 | 48.7 min |
| LightGBM | 0.5424 | 35.6 min |
| **XGBoost OvR** | **0.5500** | 69.5 min |

## Best Model: XGBoost One-vs-Rest
- **Hyperparameter tuning**: BayesSearchCV (100 iterations)
- **Class-specific boost multipliers**: [0.74, 1.11, 0.70, 0.81]
- **Improvement**: +5.7% over baseline

## Key Techniques
- Stratified 5-fold cross-validation
- Bayesian optimization for hyperparameters
- Class weight balancing
- Domain-driven feature engineering

## Conclusion
All models converged to F1 ≈ 0.52-0.55, indicating dataset limitations rather than model capacity issues. Learning curve analysis confirms sufficient data quantity but limited predictive signal in available features.
