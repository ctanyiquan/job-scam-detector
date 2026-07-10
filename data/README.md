# Data

This folder holds the datasets used by the project. `raw/` and `processed/` are
git-ignored (large and re-creatable), so they need to be set up locally before running
the notebooks.

## Getting the raw dataset

The raw dataset is the EMSCAD (Employment Scam Aegean Dataset) job postings dataset,
available on Kaggle as "Real or Fake Job Posting Prediction":

https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Download `fake_job_postings.csv` from that page and place it at:

```
data/raw/fake_job_postings.csv
```

This is the path `notebooks/01_data_understanding.ipynb`, and every notebook after it,
expect the file to be at.

## What's here

- `raw/` — the source dataset exactly as downloaded. Git-ignored.
- `processed/` — generated intermediate data (for example `cleaned.csv`), produced by
  `02_data_preparation.ipynb`. Git-ignored.
- `stress_test.csv` — a small, hand-written evaluation set. Committed, since it cannot
  be regenerated.
