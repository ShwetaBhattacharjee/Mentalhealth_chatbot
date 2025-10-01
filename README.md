This project is a **machine learning pipeline** for emotion detection in text data, combining tweets about mental health and conversation datasets. It includes **data cleaning, preprocessing, model training, evaluation, and model export**.

## Features

* Cleans text data using `neattext` and regex: removes user handles, URLs, punctuation, stopwords, and converts text to lowercase.
* Combines multiple datasets and handles **imbalanced class distributions** by filtering minority classes.
* Splits data into **training and testing sets**.

## Models

Trains and evaluates multiple machine learning models:

* **SVM (Support Vector Machine)**
* **Random Forest Classifier**
* **Gradient Boosting Classifier**

Each model is evaluated using **accuracy, precision, recall, and F1-score**, and saved as a `.pkl` file for future use.

## Technologies

* Python libraries: `pandas`, `scikit-learn`, `neattext`, `joblib`, `re`
* Models trained using **TF-IDF vectorization** for text features
* Saves trained models for deployment or further analysis

## Usage

1. Install dependencies:

```bash
pip install scikit-learn neattext pandas joblib
```

2. Run the Python script to train models and save them as `.pkl` files.
3. Models can be loaded later for predictions:

```python
import joblib
model = joblib.load('svm_model.pkl')
pred = model.predict(["Your sample text here"])
```









https://www.kaggle.com/datasets/adhamelkomy/twitter-emotion-dataset










https://www.kaggle.com/datasets/adhamelkomy/twitter-emotion-dataset
