# Third-party
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

# Local
from app import _default_classical_input_frame, classify_posting, predict_fraud_probability
from src.lstm_pipeline import LstmPipeline

TRAIN_TEXTS = ["urgent wire transfer scam free money", "great senior engineer role with benefits"]
TRAIN_LABELS = [1, 0]


def _build_classical_pipeline():
    """Fit a tiny classical pipeline (TF-IDF text + one-hot metadata) on synthetic rows, mirroring test_explain.py."""
    train_df = pd.DataFrame(
        {
            "text": TRAIN_TEXTS,
            "department": ["missing"] * len(TRAIN_TEXTS),
            "employment_type": ["Full-time"] * len(TRAIN_TEXTS),
            "required_experience": ["Mid-Senior level"] * len(TRAIN_TEXTS),
            "required_education": ["Bachelor's Degree"] * len(TRAIN_TEXTS),
            "industry": ["missing"] * len(TRAIN_TEXTS),
            "function": ["missing"] * len(TRAIN_TEXTS),
            "telecommuting": [0] * len(TRAIN_TEXTS),
            "has_company_logo": [1] * len(TRAIN_TEXTS),
            "has_questions": [0] * len(TRAIN_TEXTS),
        }
    )
    metadata_fields = ["department", "employment_type", "required_experience", "required_education", "industry", "function"]
    preprocessor = ColumnTransformer(
        transformers=[("text", TfidfVectorizer(), "text"), ("categorical", OneHotEncoder(handle_unknown="ignore"), metadata_fields)],
        remainder="passthrough",
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])
    pipeline.fit(train_df, TRAIN_LABELS)
    return pipeline


def _build_lstm_pipeline():
    """Build a tiny LstmPipeline, mirroring test_explain.py's LSTM fixture."""
    tokenizer = Tokenizer(num_words=50, oov_token="<OOV>")
    tokenizer.fit_on_texts(TRAIN_TEXTS)

    model = Sequential([Embedding(input_dim=50, output_dim=8, mask_zero=True), LSTM(4), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.build(input_shape=(None, 6))

    return LstmPipeline(tokenizer, max_length=6, model=model)


def test_default_classical_input_frame_has_expected_columns_and_defaults():
    """The classical fallback frame has all ten expected columns, with department/industry/function fixed to missing."""
    frame = _default_classical_input_frame("urgent wire transfer scam")

    expected_columns = {
        "text",
        "department",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        "telecommuting",
        "has_company_logo",
        "has_questions",
    }
    assert set(frame.columns) == expected_columns
    assert frame["department"].iloc[0] == "missing"
    assert frame["industry"].iloc[0] == "missing"
    assert frame["function"].iloc[0] == "missing"
    assert frame["text"].iloc[0] == "urgent wire transfer scam"


def test_predict_fraud_probability_uses_text_only_for_lstm_pipeline():
    """An LstmPipeline is called with the cleaned text alone, not a metadata frame."""
    lstm_pipeline = _build_lstm_pipeline()
    probability = predict_fraud_probability(lstm_pipeline, "urgent wire transfer scam")

    assert 0.0 <= probability <= 1.0


def test_predict_fraud_probability_uses_default_frame_for_classical_pipeline():
    """A classical pipeline is called with the fixed-default metadata frame and returns a valid probability."""
    classical_pipeline = _build_classical_pipeline()
    probability = predict_fraud_probability(classical_pipeline, "urgent wire transfer scam")

    assert 0.0 <= probability <= 1.0


def test_classify_posting_returns_none_for_empty_input():
    """Empty or punctuation-only input is rejected before the model is ever touched."""
    assert classify_posting(model=None, raw_text="") is None
    assert classify_posting(model=None, raw_text="!!! 123 ???") is None
