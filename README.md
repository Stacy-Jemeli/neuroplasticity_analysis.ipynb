# cerebral-palsy-neuroplasticity-prediction
Elastic-net logistic regression analysis of multidimensional functional improvement following cerebral palsy rehabilitation.
# Prediction of Optimal Functional Improvement Following Cerebral Palsy Rehabilitation

## Overview

This repository contains the Python analysis developed for the study:

**Logistic Regression Model to Predict Optimal Neuroplasticity Following Early Intervention in Cerebral Palsy**

The study used longitudinal rehabilitation data from 416 children with cerebral palsy. It developed an operational multidimensional measure of functional improvement and an elastic-net logistic regression model for predicting an optimal outcome after rehabilitation.

The term *neuroplasticity* reflects the theoretical basis of the study. The outcome used in the analysis is a composite of observed functional change and is not a direct biological measurement of neural reorganization.

## Study objectives

The study aimed to:

1. Construct and validate a multidimensional measure of functional improvement.
2. Examine the relationship between rehabilitation timing and optimal outcome.
3. Assess facility adherence, facility intensity and home-programme adherence.
4. Develop and internally validate an elastic-net logistic regression model.
5. identify stable predictors that may support individualized rehabilitation planning.

## Analytical sample

The final cleaned dataset contained:

- 416 unique participants
- 175 source variables
- 172 participants who initiated rehabilitation before 24 months
- 244 participants who initiated rehabilitation at 24 months or later
- 289 optimal outcomes
- 127 suboptimal outcomes

## Optimal Neuroplasticity Score

The Optimal Neuroplasticity Score, abbreviated as ONS, combined standardized change across six domains:

1. Gross motor function
2. Manual ability
3. Adaptive functioning
4. Communication
5. Participation
6. Health-related quality of life

The continuous ONS was compared with a functional-goal achievement anchor. Receiver operating characteristic analysis and the Youden index identified a primary threshold of −0.237.

The ONS showed:

- Cronbach's alpha: 0.778
- Anchor-validation AUC: 0.895
- Sensitivity: 0.807
- Specificity: 0.870

## Prediction model

The prediction model used elastic-net logistic regression.

The modelling procedure included:

- Median imputation for numerical predictors
- Most-frequent imputation for categorical predictors
- Standardization of numerical predictors
- One-hot encoding of categorical predictors
- Five-fold inner cross-validation
- Repeated five-fold outer cross-validation
- Three outer repetitions
- Bootstrap coefficient stability analysis
- Cross-validated calibration assessment

## Main model results

The internally validated performance was:

| Measure | Estimate |
|---|---:|
| Mean cross-validated AUC | 0.868 |
| Participant-level AUC | 0.870 |
| Mean Brier score | 0.134 |
| Mean log loss | 0.419 |
| Calibration intercept | −0.031 |
| Calibration slope | 1.048 |

The final model used:

- C: 0.464
- L1 ratio: 0.50
- Encoded features: 44
- Non-zero coefficients: 32
- Bootstrap repetitions: 300

## Main hypothesis finding

Initiation of rehabilitation before 24 months was associated with higher adjusted odds of an optimal outcome:

- Adjusted odds ratio: 10.296
- 95% confidence interval: 5.132–20.653
- p < 0.001

This is an observational association and should not be interpreted as proof of causation.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── DATA_AVAILABILITY.md
├── MODEL_CARD.md
├── ANALYSIS_PLAN.md
└── neuroplasticity_analysis.ipynb
