# SENTIMENT ANAKYSIS

This project performs sentiment analysis on textual data (tweets) using a pre-trained transformer model fine-tuned on a custom dataset. It utilizes the Hugging Face `transformers` library for loading and fine-tuning the model, and PyTorch for tensor operations.

## Overview

The script performs the following steps:

1.  **Data Loading and Preprocessing:**
    *   Loads data from a CSV file (`twitter_training.csv`).
    *   Handles missing values.
    *   Cleans the text data (removes URLs, punctuation, stopwords, and lemmatizes).
    *   Encodes sentiment labels into binary values (positive: 1, negative: 0).

2.  **Fine-tuning the Transformer Model:**
    *   Loads a pre-trained transformer model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) and its tokenizer.
    *   Creates a custom dataset class (`TweetDataset`) to handle the data in a format suitable for the `Trainer`.
    *   Splits the data into training and testing sets.
    *   Fine-tunes the pre-trained model on the training data using the `Trainer` class from `transformers`.

3.  **Model Evaluation:**
    *   Evaluates the fine-tuned model on the test data.
    *   Prints the classification report, confusion matrix, and accuracy.

4.  **Sentiment Prediction:**
    *   Provides a function to predict the sentiment of new text input using the fine-tuned model.

## Requirements

*   Python 3.7+
*   Libraries:
    *   `pandas`
    *   `numpy`
    *   `nltk`
    *   `scikit-learn`
    *   `transformers`
    *   `torch`

To install the requirements, run:

```bash
pip install pandas numpy nltk scikit-learn transformers torch accelerate
