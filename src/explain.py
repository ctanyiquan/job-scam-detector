# Third-party
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Local
from src.lstm_pipeline import LstmPipeline


def explain_prediction(model, text, top_n=10):
    """Return the words that most influenced a single prediction
    for the given cleaned text.

    `model` is either a fitted scikit-learn Pipeline (Logistic
    Regression or Linear SVM, optionally wrapped in a
    CalibratedClassifierCV) or an `LstmPipeline`. Returns a list of
    up to `top_n` (word, contribution) pairs, sorted by
    contribution magnitude. A positive contribution pushes the
    prediction toward "fraudulent"; a negative one pushes toward
    "legitimate".
    """
    if isinstance(model, LstmPipeline):
        return _explain_via_occlusion(model, text, top_n)

    if isinstance(model, CalibratedClassifierCV):
        pipeline = _unwrap_calibrated_pipeline(model)
    else:
        pipeline = model
    return _explain_via_coefficients(pipeline, text, top_n)


def _unwrap_calibrated_pipeline(calibrated_model):
    """Return the underlying fitted Pipeline a
    CalibratedClassifierCV wraps.

    The attribute holding it was renamed from `base_estimator` to
    `estimator` across scikit-learn versions, so both are checked.
    """
    calibrated_classifier = calibrated_model.calibrated_classifiers_[0]
    if hasattr(calibrated_classifier, "estimator"):
        return calibrated_classifier.estimator
    return calibrated_classifier.base_estimator


def _explain_via_coefficients(pipeline, text, top_n):
    """Explain a linear pipeline's prediction using TF-IDF weight x
    classifier coefficient per word."""
    # Pull out the fitted TF-IDF vectorizer and the classifier's
    # coefficients for the text features only
    preprocessor = pipeline.named_steps["preprocessor"]
    vectorizer = preprocessor.named_transformers_["text"]
    classifier = pipeline.named_steps["classifier"]
    text_feature_slice = preprocessor.output_indices_["text"]
    text_coefficients = classifier.coef_.ravel()[text_feature_slice]

    # Weight each word present in this text by its coefficient
    tfidf_weights = vectorizer.transform([text]).toarray().ravel()
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = np.nonzero(tfidf_weights)[0]
    contributions = [
        (feature_names[i], tfidf_weights[i] * text_coefficients[i])
        for i in nonzero_indices
    ]

    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions[:top_n]


def _explain_via_occlusion(model, text, top_n):
    """Explain an LstmPipeline's prediction by measuring how much
    removing each word shifts the fraud probability."""
    # Removing one word at a time gives every word's individual
    # effect on the prediction
    words = text.split()
    occluded_texts = [
        " ".join(words[:i] + words[i + 1 :]) for i in range(len(words))
    ]

    # Predict the baseline and every occluded variant in one batch
    # rather than one call per word
    baseline_probability = model.predict_proba([text])[0, 1]
    if occluded_texts:
        occluded_probabilities = model.predict_proba(occluded_texts)[:, 1]
    else:
        occluded_probabilities = np.array([])
    contributions = [
        (word, baseline_probability - occluded_probability)
        for word, occluded_probability in zip(words, occluded_probabilities)
    ]

    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions[:top_n]
