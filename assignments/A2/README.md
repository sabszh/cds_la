# Assignment 2: Text classification benchmarks

This project involves training a classifier on a dataset to classify news articles as either real or fake. The script contains functions for data processing, model training, and saving trained models, classification reports and carbon emission tracking.

## Data Source
The data used in this analysis can be accesed via [this link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The data contains 3 columns: title, text, label (FAKE/REAL). 

## Requirements
- Python > 3.10.12
- `codecarbon` library
- `joblib` library
- `pandas` library
- `scikit_learn` library

## Usage
1. Clone or download the repository.

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh <classifier_type>
    ```
*Replace <classifier_type> with either 'logreg' for logistic regression or 'mlp' for MLP*.

### Command-line Arguments

The script accepts the following command-line argument:

- `classifier_type`: Specifies the type of classifier to train, which can be either `'logreg'` for logistic regression or `'mlp'` for MLP classifier.

### Functions

1. **`parse_arguments()`**: Parses the command-line arguments passed to the script.
2. **`load_data(filename)`**: Loads data from a CSV file.
3. **`split_vectorize_fit_text(data, text_column, label_column, max_features, test_size, ngram_range, lowercase, max_df, min_df)`**: Splits the data into training and testing sets, vectorizes the text data using TF-IDF, and returns the necessary components for training.
4. **`train_classifier(X_train_feats, y_train, classifier_type, random_state, activation, hidden_layer_sizes, max_iter)`**: Trains a classifier on the given features and labels. It supports both logistic regression and MLP classifiers.
5. **`save_models(classifier, vectorizer, output_path, modelname)`**: Saves the trained classifier and vectorizer to disk.
6. **`save_report(y_test, y_pred, output_path)`**: Saves the classification report to a text file.

### Carbon Emission Tracker Integration
The script integrates the Carbon Emission Tracker library to monitor and track carbon emissions during classification. It includes the following functionality:

- Initializes the tracker with project details.
- Starts tracking emissions for specific tasks (e.g., data loading, model training).
- Stops tracking emissions after completing each task.
- Saves emission data to the specified output directory for further analysis.

### Main Functionality

The main function of the script orchestrates the entire process:

- It parses command-line arguments to determine the classifier type.
- Loads the data from a CSV file.
- Splits, vectorizes, and fits the text data.
- Trains the specified classifier type.
- Makes predictions on the test data.
- Saves the trained models and classification report to the output directory.

## File Structure

The project assumes the following directory structure:

```
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
<<<<<<< HEAD
```
=======
```
>>>>>>> d07bfdce3a9caa38c2fe7f3d51f23bc4afaa7555
