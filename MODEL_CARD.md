# Model Card

## Model name

Elastic-Net Logistic Regression Model for Predicting Optimal Multidimensional Functional Improvement Following Cerebral Palsy Rehabilitation

## Model status

Research model requiring external validation.

The model is not approved for independent clinical decision-making.

## Intended purpose

The model was developed to estimate the probability of an optimal multidimensional functional outcome after rehabilitation among children with cerebral palsy.

Its intended research purposes are:

- Identifying prognostic patterns
- Supporting future external validation
- Informing data-quality improvement
- Supporting research on individualized rehabilitation planning

## Target outcome

The binary outcome is derived from the Optimal Neuroplasticity Score.

The ONS combines standardized change in:

- Gross motor function
- Manual ability
- Adaptive functioning
- Communication
- Participation
- Health-related quality of life

An ONS of at least −0.237 is classified as optimal.

The ONS is a functional-change measure and not a direct biological measure of neural plasticity.

## Development data

- Participants: 416
- Optimal outcomes: 289
- Suboptimal outcomes: 127
- Early rehabilitation participants: 172
- Later rehabilitation participants: 244

## Model type

Elastic-net penalized logistic regression.

## Preprocessing

The model uses:

- Median numerical imputation
- Most-frequent categorical imputation
- Numerical standardization
- One-hot encoding
- Selected missingness indicators

All preprocessing is fitted within the cross-validation pipeline.

## Internal validation

The model was evaluated with:

- Five inner cross-validation folds
- Five outer cross-validation folds
- Three outer repetitions
- Fifteen outer-fold evaluations
- 300 bootstrap coefficient-stability repetitions

## Performance

| Measure | Estimate |
|---|---:|
| Mean outer-fold AUC | 0.868 |
| Participant-level AUC | 0.870 |
| Mean Brier score | 0.134 |
| Mean log loss | 0.419 |
| Calibration intercept | −0.031 |
| Calibration slope | 1.048 |

## Important predictive features

Stable predictive features included:

- Rehabilitation initiated before 24 months
- Facility therapy adherence
- Facility therapy intensity
- Home-programme adherence
- Missing home-adherence information
- Baseline GMFM
- GMFCS severity categories

Predictive importance does not establish causality.

## Known limitations

- Internal validation only
- No independent external cohort
- Observational design
- Potential residual confounding
- Substantial missingness in selected home-programme fields
- Inconsistency between recorded and reconstructed adherence measures
- Sample drawn from selected rehabilitation facilities
- Twelve-week outcome period
- Outcome is not a biological neural measure

## Prohibited uses

The model must not be used to:

- Deny rehabilitation
- Reduce therapy solely because of predicted probability
- Replace clinical assessment
- Replace family goals
- Make autonomous clinical decisions
- Rank the worth of children or families
- Make decisions in populations where it has not been validated

## Required next steps

Before clinical use, the model requires:

1. Source-data reconciliation
2. Prospective temporal validation
3. Geographic external validation
4. Subgroup calibration assessment
5. Fairness assessment
6. Clinical-utility evaluation
7. Governance and monitoring procedures
