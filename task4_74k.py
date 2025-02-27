import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Download NLTK resources (if needed)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load Dataset
try:
    df = pd.read_csv('Twitter_training.csv')
except FileNotFoundError:
    print("Error: 'your_data.csv' not found.")
    exit()

# Print column names and handle missing data
print("Column names:", df.columns)
df = df.dropna(subset=['Tweet', 'Sentiment'])

# --- 1. Data Exploration and Understanding ---
print("--- 1. Data Exploration ---")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Verify the unique sentiment labels
print("\nUnique Sentiment Labels:", df['Sentiment'].unique())

# --- 2. Data Preprocessing ---
print("\n--- 2. Data Preprocessing ---")

# 2.1 Text Cleaning Functions
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

df['Cleaned_Tweet'] = df['Tweet'].apply(clean_text)
df['Cleaned_Tweet'] = df['Cleaned_Tweet'].apply(remove_stopwords)
df['Cleaned_Tweet'] = df['Cleaned_Tweet'].apply(lemmatize_text)

print("\nCleaned Data:")
print(df.head())

# 2.2 Binary Encoding of Sentiment
df['Sentiment_Binary'] = df['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
print("\nBinary Encoded Sentiment:")
print(df[['Sentiment', 'Sentiment_Binary']].head())

# --- 3. Prepare Data for Fine-tuning ---
print("\n--- 3. Prepare Data for Fine-tuning ---")

# Model name
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

# Prepare dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenize the text data
encodings = tokenizer(df['Cleaned_Tweet'].tolist(), truncation=True, padding=True)
labels = df['Sentiment_Binary'].tolist()

# Create dataset
dataset = TweetDataset(encodings, labels)

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Cleaned_Tweet'].tolist(), df['Sentiment_Binary'].tolist(), test_size=0.2, random_state=42, stratify=df['Sentiment_Binary']
)

# Tokenize the split data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create the datasets
train_dataset = TweetDataset(train_encodings, train_labels)
test_dataset = TweetDataset(test_encodings, test_labels)

# --- 4. Fine-tune the Model ---
print("\n--- 4. Fine-tune the Model ---")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=16,   # Batch size per device during training
    per_device_eval_batch_size=64,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch", # Run evaluation at the end of each epoch
)

# Define trainer
trainer = Trainer(
    model=model,                         # the pretrained model to fine-tune
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

# Train the model
trainer.train()

# --- 5. Evaluate the Model ---
print("\n--- 5. Evaluate the Model ---")

# Make predictions on the test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Calculate metrics
print("Classification Report:\n", classification_report(test_labels, preds))
print("Confusion Matrix:\n", confusion_matrix(test_labels, preds))
print("Accuracy:", accuracy_score(test_labels, preds))

# --- 6. Prediction on New Data (Example) ---
print("\n--- 6. Prediction on New Data ---")

def predict_sentiment(text, tokenizer, model):
    cleaned_text = clean_text(text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = lemmatize_text(cleaned_text)

    # Tokenize the new text
    inputs = tokenizer(cleaned_text, truncation=True, padding=True, return_tensors="pt")

    # Move inputs to the GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Get probabilities

    # Get predicted class
    predicted_class = torch.argmax(predictions).item()

    return "Positive" if predicted_class == 1 else "negative"

new_comment = "This product is absolutely amazing!"
predicted_sentiment = predict_sentiment(new_comment, tokenizer, model)
print(f"The sentiment for the comment '{new_comment}' is: {predicted_sentiment}")

new_comment = "I am very upset and want a refund."
predicted_sentiment = predict_sentiment(new_comment, tokenizer, model)
print(f"The sentiment for the comment '{new_comment}' is: {predicted_sentiment}")