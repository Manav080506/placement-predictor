# Campus Placement & Salary Predictor

End-to-end ML pipeline predicting whether a student gets placed, and
what they'd earn if placed — built on the [Campus Recruitment dataset](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
(215 students, secondary school through MBA records).

## Pipeline

```
data/  ->  src/data_prep.py  ->  src/train_classification.py  ->  models/placement_classifier.pkl
                              ->  src/train_regression.py     ->  models/salary_regressor.pkl
```

## Setup

```bash
pip install -r requirements.txt
python src/data_prep.py            # cleans + encodes the raw data
python src/train_classification.py # trains placement classifier
python src/train_regression.py     # trains salary regressor
```

## Results

**Placement classification (Random Forest): 83.7% accuracy**

Top factors influencing whether a student gets placed:
1. 10th grade percentage (`ssc_p`)
2. Degree percentage (`degree_p`)
3. 12th grade percentage (`hsc_p`)
4. MBA percentage (`mba_p`)

**Salary regression: R² was negative (worse than predicting the mean)**

This is the more interesting finding, not a failure: academic performance
strongly predicts *whether* you get placed, but barely predicts *how much*
you're paid once you are. With only 148 placed students and a couple of
high-salary outliers, the data suggests salary is driven by factors this
dataset doesn't capture — company, role, negotiation — not grades. Worth
saying exactly that in an interview; it shows you read the result instead
of just reporting a number.

## Next steps

- [ ] Wrap `placement_classifier.pkl` in a small Flask/FastAPI endpoint (deploy on AWS/Azure)
- [ ] Build a Power BI dashboard on `outputs/placement_processed.csv` showing
      the placement-rate breakdown by specialization, gender, and work experience
- [ ] Link the deployed API + dashboard from the portfolio site project card

## Stack

Python, pandas, scikit-learn (Logistic Regression, Random Forest) — deep
learning skipped deliberately here since 215 rows is too small a dataset
for a neural net to beat classical ML; that's a real modeling judgment call,
not a gap.
