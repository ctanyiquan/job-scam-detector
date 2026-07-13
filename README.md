# Job Scam Detector

A machine-learning web application that classifies a pasted job posting or recruiter
message as **fraudulent** or **legitimate**, returns a confidence score, and highlights
the words that most influenced the verdict.

## How it works

Two models are trained and compared on the same held-out data: a classical model
(TF-IDF text features plus structured metadata, with Logistic Regression and Linear
SVM variants) and a deep-learning comparison model (a Bidirectional LSTM on text only).
Whichever model performs better on the held-out test set is deployed — the classical
model is favoured only as a tie-breaker, not a foregone conclusion.

The project follows the CRISP-DM process, one notebook per stage:

| Stage | Notebook | What it does |
|-------|----------|---------------|
| Data Understanding | [`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | First numeric look at the raw dataset |
| Data Preparation | [`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) | Cleans and combines text, fills missing metadata |
| Data Understanding (revisited) | [`03_eda.ipynb`](notebooks/03_eda.ipynb) | Visual exploratory analysis on the cleaned data |
| Modelling | [`04_modelling_classical.ipynb`](notebooks/04_modelling_classical.ipynb) | Tunes and validates the classical models |
| Modelling | [`05_modelling_lstm.ipynb`](notebooks/05_modelling_lstm.ipynb) | Trains and validates the LSTM |
| Evaluation | [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Compares all three, selects, tests once, deploys |

## Results

Validation-set performance (fraudulent class):

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.78 | 0.91 | 0.84 | 0.9944 |
| Linear SVM | 0.85 | 0.85 | 0.85 | 0.9927 |
| LSTM | 0.86 | 0.88 | 0.87 | 0.9803 |

The LSTM was selected (highest F1, past the tie margin) and deployed. On the held-out
test set it reached 0.82 precision, 0.82 recall, 0.82 F1, and 0.9687 ROC-AUC. Against a
small, AI-generated adversarial stress test it reached 0.82 precision but only 0.50
recall — a useful signal on where the model currently struggles, read qualitatively
given the tiny sample size, not as a formal metric.

## Project status

Notebooks 01–06 are complete and the deployed model is saved at
`models/deployed_pipeline.joblib`. The web app (`app.py`) and its hosting platform are
still being built — this section will be updated with setup and usage instructions
once that lands.

## Getting started (for the notebooks)

1. Clone the repository.
2. Download the raw dataset and place it as described in [`data/README.md`](data/README.md).
3. Create a virtual environment and install the development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```
4. Run the notebooks in order, 01 through 06, each with Restart & Run All.

## Tech stack

Python, pandas, and scikit-learn for the classical model and data pipeline;
TensorFlow/Keras for the LSTM; pytest for the test suite; joblib for the model
artifact. The web app's framework and hosting platform are not yet finalised.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branching, commit, and pull request
workflow.

## License

MIT — see [`LICENSE`](LICENSE).
