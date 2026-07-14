# Standard library
import html
import re

# Third-party
import pandas as pd

# Text fields combined into a single blob per job posting
TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def combine_fields(row, fields=TEXT_FIELDS):
    """Join the given fields of a row into one string, treating
    missing values as empty.

    `row` is a mapping of column name to value (for example, a
    pandas Series). Returns a single string with each field's text
    separated by one space.
    """
    values = [
        str(row[field]) if pd.notna(row[field]) else ""
        for field in fields
    ]
    return " ".join(values).strip()


def clean_text(text):
    """Lower-case text, unescape HTML entities, replace HTML
    tags/links with placeholders, and keep letters and spaces only.
    """
    # Normalise case
    text = text.lower()

    # Unescape HTML entities (for example &amp;) before tags are
    # stripped
    text = html.unescape(text)

    # Remove HTML tags and links, replacing them with a single space so
    # words do not fuse together
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)

    # Keep letters and single spaces only
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
