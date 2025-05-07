import string
import re
import unicodedata

ALLOWED_PUNCTUATION = " .,!?;:'\"()-\n"
ALLOWED_CHARACTERS = set(string.ascii_lowercase + string.digits + ALLOWED_PUNCTUATION)

def is_valid_char(char):
    return char.isalnum() or char in ALLOWED_PUNCTUATION

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(char for char in text if char in ALLOWED_CHARACTERS)

def normalize_v2(text):
    text = unicodedata.normalize('NFKC', text)
    return ''.join(char for char in text if is_valid_char(char))