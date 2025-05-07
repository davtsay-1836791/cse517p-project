import string
import unicodedata

ALLOWED_PUNCTUATION = " .,!?;:'\"()-\n"
ALLOWED_CHARACTERS = set(string.ascii_lowercase + string.digits + ALLOWED_PUNCTUATION)

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(char for char in text if char in ALLOWED_CHARACTERS)