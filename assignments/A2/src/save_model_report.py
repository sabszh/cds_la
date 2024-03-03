######
# Script for saving models and reports
######

# Import necessary libraires
import os
import pandas as pd
from sklearn import metrics
from joblib import dump, load

def save_models(classifier, vectorizer, output_path):
    '''
    Save trained classifier and vectorizer to disk.
    
    Args:
        classifier: Trained classifier object.
        vectorizer: Fitted vectorizer object.
        output_path (str): Path to save the models.
    '''
    dump(classifier, os.path.join(output_path, f"classifier{classifier}.joblib"))
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