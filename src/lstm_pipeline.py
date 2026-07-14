# Standard library
import os
import tempfile

# Third-party
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LstmPipeline:
    """Wrap a fitted Keras LSTM model so it can be
    joblib-serialized like a scikit-learn pipeline.

    Bundles the trained model with the Tokenizer and max sequence
    length it was trained with, and exposes predict/predict_proba
    so calling code never needs to branch on whether the deployed
    model is a scikit-learn Pipeline or this wrapper.
    """

    def __init__(self, tokenizer, max_length, model):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model

    def predict_proba(self, texts):
        """Return an (n_samples, 2) array of [P(legitimate),
        P(fraudulent)] for the given cleaned texts."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
        )
        fraud_probabilities = self.model.predict(padded).ravel()
        return np.column_stack(
            [1 - fraud_probabilities, fraud_probabilities]
        )

    def predict(self, texts):
        """Return 0/1 predictions for the given cleaned texts,
        thresholded at 0.5."""
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)

    def __getstate__(self):
        """Serialize the Keras model to bytes (native .keras
        format) so the whole wrapper survives a single
        joblib.dump()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "model.keras")
            self.model.save(temp_path)
            with open(temp_path, "rb") as model_file:
                model_bytes = model_file.read()

        return {
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
            "model_bytes": model_bytes,
        }

    def __setstate__(self, state):
        """Rebuild the Keras model from its serialized bytes on
        unpickling."""
        self.tokenizer = state["tokenizer"]
        self.max_length = state["max_length"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "model.keras")
            with open(temp_path, "wb") as model_file:
                model_file.write(state["model_bytes"])
            self.model = load_model(temp_path)
