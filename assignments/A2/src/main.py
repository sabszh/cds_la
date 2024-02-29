######
# Training logictic regression classifier on data
######

# Importing packages
# System tools
import os
import sys
sys.path.append("..")

# Data munging tools
import pandas as pd

# Machine learning stuff
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Importing local module
from vectoring import vectorizer


# Loading the data
filename = os.path.join("..","in","fake_or_real_news.csv")

data = pd.read_csv(filename, index_col=0)

vectorizer("text","label", 500)

### Fitting the data
# first we fit to the training data...
X_train_feats = vectorizer.fit_transform(X_train)

#... then do it for our test data
X_test_feats = vectorizer.transform(X_test)

# get feature names
feature_names = vectorizer.get_feature_names_out()

# Setting classifier 
classifier = LogisticRegression(random_state=9).fit(X_train_feats, y_train)

# Making predictions using classifier
y_pred = classifier.predict(X_test_feats)

# Making classification report 
class_report = metrics.classification_report(y_test, y_pred)

from joblib import dump, load
dump(classifier, "../models/LR_classifier.joblib")
dump(vectorizer, "../models/tfidf_vectorizer.joblib")

# Saving the report as a text file
# Save the report to a text file
with open("../out/classification_report.txt", "w") as report_file:
    report_file.write(class_report)

