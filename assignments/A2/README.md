# Assignment 2: Text classification benchmarks

This project involves training a classifier on a dataset to classify news articles as either real or fake. The script contains functions for data processing, model training, and saving trained models, classification reports and carbon emission tracking.

## Data Source
The data used in this analysis can be accesed via [this link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The data contains 3 columns: `title`, `text`, `label` (FAKE/REAL).

## Requirements
- Python > 3.10.12
- `codecarbon` library
- `setuptools` library
- `joblib` library
- `pandas` library
- `scikit_learn` library

## Usage
To use this script, follow these steps:

1. Clone or download the repository and make sure you have the file structure as pointed out, and the needed files stored in `in`

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh <classifier_type>
    ```
    - `<classifier_type>`: Model to classify with, either 'logreg' for logistic regression or 'mlp' for MLP*.

## Script Overview
This Python script is designed for conducting text classification benchmarks. It utilizes machine learning classifiers: logistic regression and multilayer perceptron (MLP) to train models on text data and evaluate their performance. Below is a brief overview of the functionalities provided by the script:

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

## Output Summary
The output of the text classification benchmarks is presented below for both logistic regression (logreg) and multilayer perceptron (MLP) classifiers.

### Table 1: For logistic regression (logreg)
```
              precision    recall  f1-score   support

        FAKE       0.89      0.88      0.89       628
        REAL       0.88      0.90      0.89       639

    accuracy                           0.89      1267
   macro avg       0.89      0.89      0.89      1267
weighted avg       0.89      0.89      0.89      1267
```

### Table 2: For multilayer perceptron (MLP)
```
              precision    recall  f1-score   support

        FAKE       0.90      0.87      0.88       628
        REAL       0.87      0.90      0.89       639

    accuracy                           0.89      1267
   macro avg       0.89      0.89      0.89      1267
weighted avg       0.89      0.89      0.89      1267
```
Both models achieve around 89% accuracy in classifying fake and real news articles. Precision, recall, and F1-score provide insights into how effectively each model identifies true instances and minimizes false positives or negatives. Despite their strong performance, further optimization could address challenges like class imbalance.

## Discussion of Limitations and Possible Steps to Improvement
While the classifiers achieved reasonably high accuracy, there are several limitations and potential areas for improvement: Gaining insights into the nature of the data would be an initial step towards understanding the model's overall performance. The manner in which the fake dataset was generated or collected remains undisclosed, making it challenging to assert the efficacy of the models or evaluate the fidelity of the data representation. Testing the model on another dataset, could be one way to gain further understanding. Additionally, including a separate validation set during model training could provide a more reliable estimate of performance and help prevent overfitting to the training data.

A significant limitation of the benchmark lies in its exclusive reliance on TF-IDF vectorization for feature extraction. This method adopts a bag-of-words approach, which assumes that the occurrence of words in the document is independent of each other, disregarding the contextual relationships between them. This oversimplification may overlook crucial semantic nuances present in the text data, potentially constraining the model's ability to capture complex patterns and meanings.

Moreover, the classifiers' hyperparameters, such as the number of hidden layers and neurons for the MLP classifier, were not extensively tuned. Conducting a thorough hyperparameter search using techniques like grid search or randomized search could yield better-performing and more accurate models.

Taking a further stride, enhancing the interpretability of the models could provide insights into the features driving the predictions. This would make the models useful in terms of understanding the patterns of fake / real news.

## CodeCarbon Tracking
To track emissions, the script utilizes CodeCarbon. Emission data for each task is recorded in a CSV files located in the `out` directory.

For a more detailed analysis of these results, please see Assignment 5.

## File Structure

The project assumes the following directory structure:

```
.
A2/
│
├── in/
│ └── fake_or_real_news.csv
│
├── out/
│ ├── emissions/
│ │  ├── emissions_base_{UUID}.csv # Check the task_name for which classifier type 
│ │  ├── emissions_base_{UUID}.csv # As stated above
│ │  └── emissions.csv # This file should just be ignored
│ ├── models/
│ │  ├── classifier_logreg.joblib
│ │  ├── classifier_mlp.joblib
│ │  └── vectorizer.joblib
│ ├── logreg_report.txt
│ └── mlp_report.txt
│
├── src/
│   └── text_classification.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```