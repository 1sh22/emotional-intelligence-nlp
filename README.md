# Emotion Classification using Machine Learning and Deep Learning

This project focuses on classifying text-based emotional expressions into categories like joy, anger, sadness, love, etc. It applies both traditional machine learning models and deep learning techniques to perform multi-class classification using natural language processing.

## Overview

Given a dataset of comments labeled with emotions, the goal is to build a model that can accurately predict the emotion behind unseen text.

Key components:
- Data preprocessing and cleaning
- NLP techniques: tokenization, stemming, stopword removal
- Feature extraction using TF-IDF
- Multi-model training: Random Forest, SVM, Logistic Regression, Naive Bayes
- Deep learning with LSTM (Keras)
- Model evaluation using accuracy and classification reports
- Custom prediction function for real-time text input

## Models Used

- **Traditional Machine Learning Models**: 
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Multinomial Naive Bayes
  - Random Forest

- **Deep Learning**:
  - LSTM using Keras with embeddings and dropout

## Results

- Highest accuracy using Random Forest (~85%)
- LSTM model achieved over 97% training accuracy with early stopping

## File Structure

- `emotional_intelligence_nlp.ipynb`: Full Colab notebook with all steps
- `logistic_regression.pkl`, `label_encoder.pkl`, `tfidf_vectorizer.pkl`: Saved model components
- `README.md`: Project description

## How to Use

1. Clone the repo
2. Run the Jupyter/Colab notebook
3. Train models or load pretrained ones
4. Use `predict_emotion()` to classify new text inputs

## Future Improvements

- Integrate BERT or transformer-based models for better generalization
- Deploy as a web app using Streamlit or Flask 
- Extend emotion categories and include multilingual datasets

## Requirements

- Python 3.8+
- Libraries: `sklearn`, `keras`, `tensorflow`, `nltk`, `seaborn`, `matplotlib`, `wordcloud`, `pandas`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
