# Third-party
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

# Local
from src.explain import explain_prediction
from src.lstm_pipeline import LstmPipeline

# A handful of synthetic rows, deliberately small, mirroring the shape a real Pipeline is fitted on: a
# "text" column, one categorical column, and one passthrough flag column
TRAIN_TEXTS = [
    "urgent wire transfer scam free money",
    "great senior engineer role with benefits",
    "wire transfer scam urgent payment",
    "engineer role with benefits and team",
    "urgent wire money scam free payment",
    "team engineer benefits role senior",
]
TRAIN_LABELS = [1, 0, 1, 0, 1, 0]


def _build_pipeline(classifier):
    """Fit a tiny classical pipeline (TF-IDF text + one-hot category) on the synthetic training rows."""
    train_df = pd.DataFrame({"text": TRAIN_TEXTS, "department": ["sales"] * len(TRAIN_TEXTS), "flag": [0] * len(TRAIN_TEXTS)})
    preprocessor = ColumnTransformer(
        transformers=[("text", TfidfVectorizer(), "text"), ("categorical", OneHotEncoder(handle_unknown="ignore"), ["department"])],
        remainder="passthrough",
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(train_df, TRAIN_LABELS)
    return pipeline


def test_explain_prediction_returns_word_contributions_for_logistic_regression():
    """A fitted Logistic Regression pipeline's fraud-associated words come back with a positive contribution."""
    pipeline = _build_pipeline(LogisticRegression(max_iter=1000))
    contributions = explain_prediction(pipeline, "urgent wire transfer scam", top_n=5)

    words = [word for word, _ in contributions]
    assert len(contributions) <= 5
    assert "scam" in words


def test_explain_prediction_unwraps_calibrated_svm():
    """A LinearSVC pipeline wrapped in CalibratedClassifierCV is unwrapped to reach its coefficients."""
    svm_pipeline = _build_pipeline(LinearSVC(max_iter=5000))
    # cv=2 keeps this workable on the tiny six-row synthetic set; a real validation set is large enough for the default
    calibrated_svm = CalibratedClassifierCV(FrozenEstimator(svm_pipeline), method="sigmoid", cv=2)
    calibrated_svm.fit(pd.DataFrame({"text": TRAIN_TEXTS, "department": ["sales"] * len(TRAIN_TEXTS), "flag": [0] * len(TRAIN_TEXTS)}), TRAIN_LABELS)

    contributions = explain_prediction(calibrated_svm, "urgent wire transfer scam", top_n=5)

    words = [word for word, _ in contributions]
    assert len(contributions) <= 5
    assert "scam" in words


def test_explain_prediction_uses_occlusion_for_lstm_pipeline():
    """An LstmPipeline's prediction is explained by occlusion, returning one contribution per word."""
    tokenizer = Tokenizer(num_words=50, oov_token="<OOV>")
    tokenizer.fit_on_texts(TRAIN_TEXTS)

    model = Sequential([Embedding(input_dim=50, output_dim=8, mask_zero=True), LSTM(4), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.build(input_shape=(None, 6))

    lstm_pipeline = LstmPipeline(tokenizer, max_length=6, model=model)
    text = "urgent wire transfer scam free"
    contributions = explain_prediction(lstm_pipeline, text, top_n=5)

    assert len(contributions) == min(5, len(text.split()))
    assert all(isinstance(word, str) and isinstance(contribution, (int, float, np.floating)) for word, contribution in contributions)
