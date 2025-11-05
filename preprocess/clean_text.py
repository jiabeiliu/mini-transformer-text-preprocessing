# clean_text.py
import re

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-alphanumeric symbols except basic punctuation
    text = re.sub(r'[^a-z0-9.,!? ]+', '', text)
    # Strip leading/trailing spaces
    return text.strip()
