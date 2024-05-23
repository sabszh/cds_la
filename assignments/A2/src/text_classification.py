"""
Assignment: 2 - Text classification benchmarks
Course: Language Analytics
Author: Sabrina Zaki Hansen
"""

# Importing necessary packages
import os
import sys
import argparse
import pandas as pd
sys.path.append("..")
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump, load
from codecarbon import EmissionsTracker

######
# Defining functions
######

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train classifiers on data.")
    parser.add_argument('classifier_type', type=str, choices=['logreg', 'mlp'], help="Type of classifier to train (logreg or mlp)")
    args = parser.parse_args()
    return args

def load_data(filename):
    '''
    Load data from a CSV file.
    
    Args:
        filename (str): Path to the CSV file.
        
    Returns:
        DataFrame: Loaded data.
    '''
    data = pd.read_csv(filename, index_col = 0)
    return data

def split_vectorize_fit_text(data, text_column, label_column, max_features, test_size=0.2, ngram_range=(1, 2), lowercase=True, max_df=0.95, min_df=0.05):
    '''
    This function splits data and vectorizes text.
    
    Args:
        data (DataFrame): Input data containing text and labels.
        text_column (str): Name of the column containing text.
        label_column (str): Name of the column containing labels.
        max_features (int): Maximum number of features to be considered.
        test_size (float): Size of the test data. Default is 0.2.
        ngram_range (tuple): Range for ngrams. Default is (1, 2).
        lowercase (bool): Convert text to lowercase. Default is True.
        max_df (float or int): Maximum document frequency for the TF-IDF. Default is 0.95.
        min_df (float or int): Minimum document frequency for the TF-IDF. Default is 0.05.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, vectorizer
    '''
    # Extracting text and labels from the data
    X = data[text_column]
    y = data[label_column]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initializing TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                lowercase=lowercase,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features)

    # Vectorizing text
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    return X_train_feats, X_test_feats, y_train, y_test, vectorizer

# Script for model training
def train_classifier(X_train_feats, y_train, classifier_type='logreg', random_state=42, activation='logistic', hidden_layer_sizes=(20,), max_iter=1000):
    '''
    Train a classifier on the given features and labels.
    
    Args:
        X_train_feats (array-like): Features for training.
        y_train (array-like): Labels for training.
        classifier_type (str): Type of classifier to use. Default is 'logreg'.
        random_state (int): Random state for reproducibility. Default is 42.
        activation (str): Activation function for MLP classifier. Default is 'logistic'.
        hidden_layer_sizes (tuple): Number of neurons in each hidden layer for MLP classifier. Default is (20,).
        max_iter (int): Maximum number of iterations for MLP classifier. Default is 1000.
        
    Returns:
        Trained classifier.
    '''
    if classifier_type == 'logreg':
        classifier = LogisticRegression(random_state=random_state)
    elif classifier_type == 'mlp':
        classifier = MLPClassifier(activation=activation,
                                   hidden_layer_sizes=hidden_layer_sizes,
                                   max_iter=max_iter,
                                   random_state=random_state)
    else:
        raise ValueError("Invalid classifier type. Choose 'logreg' or 'mlp'.")

    fit_classifier = classifier.fit(X_train_feats, y_train)
    return fit_classifier

# Script for saving models and reports
def save_models(classifier, vectorizer, output_path,modelname):
    '''
    Save trained classifier and vectorizer to disk.
    
    Args:
        classifier: Trained classifier object.
        vectorizer: Fitted vectorizer object.
        output_path (str): Path to save the models.
    '''
    dump(classifier, os.path.join(output_path, f"classifier_{modelname}.joblib"))
    dump(vectorizer, os.path.join(output_path, "vectorizer.joblib"))

def save_report(y_test, y_pred, output_path):
    '''
    Save classification report to a text file.
    
    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_path (str): Path to save the report.
    '''
    class_report = metrics.classification_report(y_test, y_pred)
    
    with open(output_path, "w") as report_file:
        report_file.write(class_report)
    print("Report file saved")

######
# Main function
######

def main():
    """
    Main function to execute the script.
    """
    # Parse the arguments
    args = parse_arguments()

    # Create out directory if it does not exist
    if not os.path.exists(os.path.join("out","emissions")):
        os.makedirs(os.path.join("out","emissions"))

    # Start CodeCarbon tracker
    tracker = EmissionsTracker(project_name="Text Classification", 
                              experiment_id="text_classifier",
                              output_dir = os.path.join("out" , "emissions"))

    # Loading the data
    tracker.start_task("load_data")
    data = load_data(os.path.join("in", "fake_or_real_news.csv"))
    tracker.stop_task()

    # Splitting, vectorizing, and fitting the text data
    tracker.start_task("split_vectorize_fit_text")
    X_train_feats, X_test_feats, y_train, y_test, vectorizer = split_vectorize_fit_text(data, "text", "label", 500)
    tracker.stop_task()

    if args.classifier_type == 'logreg':
        # Training logistic regression classifier
        tracker.start_task("train_classifier_logreg")
        classifier = train_classifier(X_train_feats, y_train, classifier_type='logreg')
        tracker.stop_task()
    elif args.classifier_type == 'mlp':
        # Training MLP classifier
        tracker.start_task("train_classifier_mlp")
        classifier = train_classifier(X_train_feats, y_train, classifier_type='mlp')
        tracker.stop_task()
    else:
        raise ValueError("Invalid classifier type. Choose between 'logreg' and 'mlp'.")

    # Making predictions
    tracker.start_task("predict")
    y_pred = classifier.predict(X_test_feats)
    tracker.stop_task()

    # Saving model
    ## Create out directory if it does not exist
    if not os.path.exists('out/models'):
        os.makedirs('out/models')
    
    tracker.start_task("save_models")
    save_models(classifier, vectorizer, os.path.join("out/models"), args.classifier_type)
    tracker.stop_task()

    # Saving classification report
    tracker.start_task("save_report")
    save_report(y_test, y_pred, os.path.join("out", f"{args.classifier_type}_report.txt"))
    tracker.stop_task()

    _ = tracker.stop()

if __name__ == "__main__":    
    main()