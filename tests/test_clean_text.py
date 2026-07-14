# Third-party
import numpy as np
import pandas as pd

# Local
from src.clean_text import clean_text, combine_fields


def test_clean_text_lower_cases():
    """Upper-case letters are folded to lower-case."""
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_strips_html_tags():
    """HTML tags are replaced with a space, not removed outright,
    so words do not fuse together."""
    assert clean_text("<p>Hello</p><p>World</p>") == "hello world"


def test_clean_text_unescapes_html_entities():
    """HTML entities such as &amp; are unescaped before
    punctuation is stripped."""
    assert clean_text("Sales &amp; Marketing") == "sales marketing"


def test_clean_text_replaces_links():
    """Links are replaced with the placeholder token url."""
    assert clean_text("Visit https://example.com now") == "visit url now"


def test_clean_text_removes_punctuation():
    """Punctuation and digits are removed, leaving letters and
    single spaces only."""
    cleaned = clean_text("Great job, apply now! Earn $500/day.")
    assert cleaned == "great job apply now earn day"


def test_clean_text_handles_empty_string():
    """An empty string cleans to an empty string."""
    assert clean_text("") == ""


def test_combine_fields_joins_present_values():
    """Present fields are joined with a single space, in the given
    field order."""
    row = pd.Series(
        {"title": "Analyst", "company_profile": "A great company"}
    )
    combined = combine_fields(row, fields=["title", "company_profile"])
    assert combined == "Analyst A great company"


def test_combine_fields_treats_missing_as_empty():
    """Missing values (NaN) contribute no text and no extra spaces
    are left behind."""
    row = pd.Series({"title": "Analyst", "company_profile": np.nan})
    combined = combine_fields(row, fields=["title", "company_profile"])
    assert combined == "Analyst"
