#!/usr/bin/python

"""
Main script for training classifier on data.
"""

# Importing necessary packages
import os
import sys
import argparse
sys.path.append("..")  # To import local modules

# Importing local modules
from data_processing import load_data, split_vectorize_fit_text
from model_training import train_classifier
from save_model_report import save_models, save_report

def main(classifier_type):
    # Loading the data
    data = load_data(os.path.join("in", "fake_or_real_news.csv"))

    # Splitting, vectorizing, and fitting the text data
    X_train_feats, X_test_feats, y_train, y_test, vectorizer = split_vectorize_fit_text(data, "text", "label", 500)

    if classifier_type == 'logreg':
        # Training logistic regression classifier
        classifier = train_classifier(X_train_feats, y_train, classifier_type='logreg')
    elif classifier_type == 'mlp':
        # Training MLP classifier
        classifier = train_classifier(X_train_feats, y_train, classifier_type='mlp')
    else:
        raise ValueError("Invalid classifier type. Choose between 'logreg' and 'mlp'.")

    # Making predictions
    y_pred = classifier.predict(X_test_feats)

    # Saving model
    save_models(classifier, vectorizer, os.path.join("models"), classifier_type)

    # Saving classification report
    save_report(y_test, y_pred, os.path.join("out", f"{classifier_type}_report.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifiers on data.")
    parser.add_argument('classifier_type', type=str, choices=['logreg', 'mlp'], help="Type of classifier to train (logreg or mlp)")
    args = parser.parse_args()
    main(args.classifier_type)