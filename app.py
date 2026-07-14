# Third-party
import joblib
import pandas as pd
import streamlit as st

# Local
from src.clean_text import clean_text
from src.explain import explain_prediction
from src.lstm_pipeline import LstmPipeline

# Three examples from data/stress_test.csv, hardcoded here so the app never reads that
# file at runtime (its role is evaluation, not app serving). Example 2 is deliberately
# paired with example 1's job title/seniority so the demo contrast isn't just topic-driven.
SAMPLE_GALLERY = [
    {
        "label": "Clear legitimate posting",
        "text": (
            "Meridian Software is hiring a Senior Software Engineer (remote, US). You'll design and "
            "maintain backend services in Go and Python and collaborate across teams. Requirements: 5+ "
            "years of experience and a CS degree or equivalent. Apply through our careers page; the "
            "process includes a recruiter screen, a technical interview, and a system-design round. We "
            "are an equal opportunity employer."
        ),
    },
    {
        "label": "Sophisticated scam",
        "text": (
            "Hi, I'm a technical recruiter at NorthBridge Labs and came across your profile. We're hiring "
            "a fully remote Senior Software Engineer at $220k with no interview panel needed. To move "
            "forward, please complete our take-home by cloning our private assessment repo and running "
            "the included setup.sh to configure the environment, then message me on Telegram to continue."
        ),
    },
    {
        "label": "Obvious scam",
        "text": (
            "Work-from-home Data Analyst needed. Earn $4,500/week processing simple spreadsheets. No "
            "experience required and no interview. To get started, a one-time onboarding fee of $150 is "
            "required for your software license, refundable after your first month."
        ),
    },
]


@st.cache_resource
def load_pipeline():
    """Load the deployed pipeline artifact once per server process."""
    return joblib.load("models/deployed_pipeline.joblib")


def _default_classical_input_frame(cleaned_text):
    """Return the one-row DataFrame a classical ColumnTransformer would need.

    Only reached if `deployed_pipeline.joblib` is ever swapped for a classical rebuild —
    the currently deployed LSTM never calls this, since it only ever sees `text`. Metadata
    is not collected from the UI, so every field here is a fixed, training-time default
    (sourced from data/processed/cleaned.csv value counts) rather than a real answer.
    """
    return pd.DataFrame(
        {
            "text": [cleaned_text],
            "department": ["missing"],
            "employment_type": ["Full-time"],
            "required_experience": ["Mid-Senior level"],
            "required_education": ["Bachelor's Degree"],
            "industry": ["missing"],
            "function": ["missing"],
            "telecommuting": [0],
            "has_company_logo": [1],
            "has_questions": [0],
        }
    )


def predict_fraud_probability(model, cleaned_text):
    """Return the fraud probability for one cleaned posting, adapting to what the loaded model expects."""
    if isinstance(model, LstmPipeline):
        return model.predict_proba([cleaned_text])[0, 1]
    return model.predict_proba(_default_classical_input_frame(cleaned_text))[0, 1]


def classify_posting(model, raw_text, top_n=10):
    """Clean the pasted text, predict its fraud probability, and return the verdict, confidence, and top words.

    Returns None if the text cleans to nothing, so the caller can show a validation
    message instead of running a degenerate all-empty input through the model.
    """
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        return None

    fraud_probability = predict_fraud_probability(model, cleaned_text)
    verdict = "Fraudulent" if fraud_probability >= 0.5 else "Legitimate"
    contributions = explain_prediction(model, cleaned_text, top_n=top_n)
    return verdict, fraud_probability, contributions


def render_result(verdict, fraud_probability, contributions):
    """Display the verdict, confidence in that verdict, and the top contributing words by sign."""
    # Show confidence in the displayed verdict, not always raw P(fraudulent)
    confidence = fraud_probability if verdict == "Fraudulent" else 1 - fraud_probability

    if verdict == "Fraudulent":
        st.error(f"Likely Fraudulent - {confidence:.1%} confidence")
    else:
        st.success(f"Likely Legitimate - {confidence:.1%} confidence")

    # Split contributions by sign: positive pushes toward fraudulent, negative toward legitimate
    toward_fraudulent = [(word, value) for word, value in contributions if value > 0]
    toward_legitimate = [(word, value) for word, value in contributions if value < 0]

    fraud_column, legit_column = st.columns(2)
    with fraud_column:
        st.markdown("**Pushed toward fraudulent**")
        for word, value in toward_fraudulent:
            st.write(f"{word} ({value:+.4f})")
    with legit_column:
        st.markdown("**Pushed toward legitimate**")
        for word, value in toward_legitimate:
            st.write(f"{word} ({value:+.4f})")


def main():
    """Lay out the page: title, sample gallery, text input, and the classify button."""
    # Page setup
    st.set_page_config(page_title="Job Scam Detector", page_icon=":mag:")
    st.title("Job Scam Detector")
    st.write("Paste a job posting or recruiter message to check whether it looks fraudulent or legitimate.")

    # Sample gallery: clicking a button sets session_state before the text_area widget
    # below is created in this same script pass, so the widget picks it up without a
    # value= conflict
    st.write("Try a sample:")
    gallery_columns = st.columns(len(SAMPLE_GALLERY))
    for column, sample in zip(gallery_columns, SAMPLE_GALLERY):
        with column:
            if st.button(sample["label"]):
                st.session_state["posting_text"] = sample["text"]

    # Pasted-text input
    posting_text = st.text_area("Job posting or recruiter message", key="posting_text", height=200)

    # Run the classifier and render the result
    if st.button("Check this posting", type="primary"):
        model = load_pipeline()
        result = classify_posting(model, posting_text)
        if result is None:
            st.warning("Please paste a job posting or recruiter message first.")
        else:
            verdict, fraud_probability, contributions = result
            render_result(verdict, fraud_probability, contributions)


if __name__ == "__main__":
    main()
