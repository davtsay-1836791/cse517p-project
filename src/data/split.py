from datasets import load_dataset
import re
import unicodedata
from langdetect import detect
import ast

# Load the entire CSV file
dataset = load_dataset("csv", data_files="src/data/mldd_dataset.csv")

# Split the dataset into train and validation sets (90% train, 10% validation)
train_dataset, dev_dataset = dataset["train"].train_test_split(test_size=0.1).values()

# Function to normalize text
def normalize_value(text):
    # Remove leading/trailing spaces
    text = text.strip()
    
    # Remove special characters and unwanted punctuation
    text = re.sub(r'[^A-Za-z0-9áéíóúàèìòùäëïöüâêîôûãõÇçÁÉÍÓÚ]+', ' ', text)
    
    # Normalize text: converting unicode characters (accents) to standard characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Optionally, you can use a language detection library to handle language-specific normalizations
    try:
        lang = detect(text)
        # If necessary, perform language-specific processing here based on detected language.
        # Example: you can exclude non-text elements or special characters for different languages
    except:
        lang = "unknown"
    
    return text, lang

def normalize_conversations(conversation):
    normalized = []
    # Normalize each entry in the conversation
    for entry in conversation:
        # Extract the text (value) to normalize
        text = entry["value"]
        
        # Normalize the text
        normalized_text, lang = normalize_value(text)
        
        # Store the normalized data
        normalized.append({
            "normalized": normalized_text,  # Normalized version
            "language": lang        # Detected language (optional)
        })
    
    return normalized


normalized_train_data = []
normalized_dev_data = []

# Parse the conversation string into a list of dictionaries
train_conversations = ast.literal_eval(train_dataset['conversations'])
dev_conversations = ast.literal_eval(dev_dataset['conversations'])

# Prepare a list to store normalized data
normalized_train_conversations = normalize_conversations(train_conversations)
normalized_dev_conversations = normalize_conversations(dev_conversations)