"""

Drug-Interaction-NLP-CDS.ipynb

Original file is located at
    https://colab.research.google.com/drive/1UrJ51Amf8SQ-hMtH_HDyUdzyyYGZ3wdv

"""

import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy

# Load the merged dataset
merged_data = pd.read_csv('./structured_medical_data.csv')

# Create a dictionary to store unique values for each column
unique_values = {}

# Loop through all columns except 'PATIENT'
for column in merged_data.columns:
    if column != 'PATIENT':
        # Flatten lists if column contains lists, otherwise just find unique values
        if merged_data[column].dropna().apply(lambda x: isinstance(x, str)).all():
            unique_values[column] = merged_data[column].dropna().unique().tolist()
        else:
            # Extract unique elements from lists
            unique_items = set()
            merged_data[column].dropna().apply(lambda x: unique_items.update(eval(x) if isinstance(x, str) else x))
            unique_values[column] = list(unique_items)

# Save unique values as a JSON or CSV file for further analysis
unique_values_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values.items()]))
unique_values_df.to_csv('./unique_column_values.csv', index=False)

# Display unique possibilities for each column
for column, values in unique_values.items():
    print(f"Column: {column}\nUnique Values: {values}\n")

# Ensure required NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load NLP models
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")  # General-purpose NLP model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bio_bert_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
ner_pipeline = pipeline("ner", model=bio_bert_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load the dataset
mtsamples = pd.read_csv('./mtsamples.csv')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

# Function to split text into chunks for BioBERT with truncation
def split_text_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunk_size = max_length - 2  # Reserve space for [CLS] and [SEP] tokens
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        yield tokenizer.convert_tokens_to_string(chunk_tokens)

# Function to extract features
def extract_features(text):
    text = preprocess_text(text)
    doc = nlp(text)

    structured_data = {
        "Medications": [],
        "Conditions": [],
        "Procedures": [],
        "Devices": [],
        "Allergies": [],
        "Sentiment": None
    }

    # SciSpacy NER
    for entity in doc.ents:
        label = entity.label_.lower()
        if "drug" in label or "medication" in label:
            structured_data["Medications"].append(entity.text)
        elif "disease" in label or "condition" in label:
            structured_data["Conditions"].append(entity.text)
        elif "procedure" in label:
            structured_data["Procedures"].append(entity.text)
        elif "device" in label:
            structured_data["Devices"].append(entity.text)
        elif "allergy" in label:
            structured_data["Allergies"].append(entity.text)

    # BioBERT NER with chunk handling
    for chunk in split_text_into_chunks(text, tokenizer):
        bio_ner_results = ner_pipeline(chunk)
        for result in bio_ner_results:
            label = result["entity_group"].lower()
            entity = result["word"]
            if "medication" in label or "drug" in label:
                structured_data["Medications"].append(entity)
            elif "disease" in label or "condition" in label:
                structured_data["Conditions"].append(entity)
            elif "procedure" in label:
                structured_data["Procedures"].append(entity)
            elif "device" in label:
                structured_data["Devices"].append(entity)
            elif "allergy" in label:
                structured_data["Allergies"].append(entity)

    # Deduplicate extracted features
    for key in structured_data.keys():
        if key != "Sentiment":
            structured_data[key] = list(set(structured_data[key]))

    # Sentiment Analysis
    sentiment = sentiment_analyzer.polarity_scores(text)
    structured_data["Sentiment"] = sentiment["compound"]

    return structured_data

# Apply feature extraction to all rows in mtsamples
structured_features = []
for i, row in mtsamples.iterrows():
    if 'transcription' not in row or pd.isna(row['transcription']):
        continue
    features = extract_features(row['transcription'])
    structured_features.append(features)
    if i == 10:  # Process only the first 10 rows for testing
        break

# Convert extracted features into a DataFrame
final_df = pd.DataFrame(structured_features)

# Merge with original mtsamples data if needed
final_output = pd.concat([mtsamples.iloc[:len(final_df)].reset_index(drop=True), final_df], axis=1)

# Save to CSV
final_output.to_csv('/content/structured_mtsamples.csv', index=False)

print("Feature extraction complete. Results saved to /content/structured_mtsamples.csv")

files = {
    "/content/ADMISSIONS.csv": ["DIAGNOSIS", "HAS_CHARTEVENTS_DATA"],
    "/content/CHARTEVENTS.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": [],
    "/content/.csv": []
}
common_identifier = "ROW_ID"
