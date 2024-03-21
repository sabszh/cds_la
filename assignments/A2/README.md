# Text classification benchmarks

This project involves training a classifier on a dataset to classify news articles as either real or fake. It includes scripts for data processing, model training, and saving trained models and classification reports.

## Usage

### 1. Main script for training classifier on data

- **Description**: This script loads the data, preprocesses it, trains two classifiers (Logistic Regression and MLP), makes predictions, and saves the trained models and classification reports.

- **Usage**:
  ```python
  python main.py
  ```

### 2. Script for data processing functions

- **Description**: This script contains functions for loading data, splitting it into training and testing sets, vectorizing text data using TF-IDF, and preparing features and labels for training.

- **Functions**:
  - `load_data(filename)`: Load data from a CSV file.
  - `split_vectorize_fit_text(data, text_column, label_column, max_features, test_size=0.2, ngram_range=(1, 2), lowercase=True, max_df=0.95, min_df=0.05)`: Split data, vectorize text, and prepare features and labels.

### 3. Script for model training

- **Description**: This script contains functions for training classifiers (Logistic Regression and MLP) using the provided features and labels.

- **Functions**:
  - `train_classifier(X_train_feats, y_train, classifier_type='logreg', random_state=42, activation='logistic', hidden_layer_sizes=(20,), max_iter=1000)`: Train a classifier on the given features and labels.

### 4. Script for saving models and reports

- **Description**: This script includes functions for saving trained classifier models and classification reports to disk.

- **Functions**:
  - `save_models(classifier, vectorizer, output_path)`: Save trained classifier and vectorizer to disk.
  - `save_report(y_test, y_pred, output_path)`: Save classification report to a text file.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- joblib

## File Structure

The project assumes the following directory structure:

```
.
project_root/
│
├── in/
│ └── fake_or_real_news.csv
│
├── out/
│ ├── logreg_report.txt
│ └── mlp_report.txt
│
├── models/
│ ├── classifierLogisticRegression.joblib
│ ├── classifierMLPClassifier.joblib
│ └── vectorizer.joblib
│
├── src/
│ ├── data_processing.py
│ ├── model_training.py
│ ├── save_model_report.py
│ └── main_script.py
│
├── requirements.txt
└── README.md
```
