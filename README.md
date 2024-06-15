# Fake News Detector

The Fake News Detector is a machine learning project designed to identify and classify news articles as either real or fake using natural language processing (NLP) techniques. This project leverages a variety of data preprocessing methods and classification algorithms to achieve accurate predictions.


## Usage

1. Ensure `train.csv` and `test.csv` are in the project directory.
2. Run the script:
    ```sh
    python app.py
    ```

## Data

The dataset includes `title`, `author`, `text`, and `label` (1 for real, 0 for fake).

## Preprocessing

1. Fill missing values.
2. Merge `title`, `author`, and `text` into `total`.
3. Clean text using regex.
4. Tokenize, remove stopwords, and lemmatize.

## Modeling

Uses Logistic Regression, Random Forest, and Decision Tree classifiers with TF-IDF vectorization.

## Pipeline

A Scikit-Learn pipeline automates preprocessing and modeling steps. The pipeline is saved as `pipeline.sav`.

## Prediction

Load and use the saved pipeline:
```python
import joblib
loaded_model = joblib.load('pipeline.sav')
text = ['Your news article text here']
result = loaded_model.predict(text)
print(result)
