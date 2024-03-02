######
# Training logictic regression classifier on data
######

# Importing system packages
import os
import sys
sys.path.append("..")

# Importing other relevant packages
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump, load

# Importing local module
from data_processing import load_data, split_vectorize_fit_text
from model_training import train_classifier
from save_model_report import save_models, save_report

# Loading the data
data = load_data(os.path.join("in","fake_or_real_news.csv"))

X_train_feats, X_test_feats, y_train, y_test, vectorizer = split_vectorize_fit_text(data,"text","label", 500)

# Train classifier and get predictions
classifier, y_pred = train_classifier(X_train, y_train, 'logistic_regression')

# Saving models
save_models(classifier, vectorizer,os.path.join("models"))

# Making classification report 
save_report(y_test,y_pred,os.path.join("out"))