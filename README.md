# Exploratory Data Analysis - Breast Cancer Wisconsin (Diagnostic)

Week 2 task of the Yuva Intern *Data Science with Python* virtual internship.

## Goal
Explore a public dataset with Pandas, Matplotlib and Seaborn: summarize it,
transform/aggregate it, visualize patterns, and interpret the results.

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (569 samples, 30 numeric
features + diagnosis). Loaded via `sklearn.datasets.load_breast_cancer`.

## What the script does
1. Loads data and checks shape, dtypes, missing values, duplicates
2. Summary statistics (mean, std, median, skew) for the 10 "mean" features
3. Transformations: readable labels, group means, IQR outlier counts, quartile size groups
4. Seven annotated charts saved to `figures/`

## Run
```
pip install -r requirements.txt
python eda_breast_cancer.py
```

## Key findings
- 357 benign vs 212 malignant; no missing values or duplicates
- Malignant samples: concavity ~3.5x and area ~2.1x higher on average
- Radius, perimeter, area almost perfectly correlated (r > 0.98)
- Malignancy rate rises from 2.1% to 95.8% across mean-radius quartiles
- Fractal dimension shows no relationship with diagnosis

Full write-up: `Week2_EDA_Report.docx`
