######
# Main script for training classifier on data
######

# Importing necessary packages
import os
import sys
sys.path.append("..")  # To import local modules

# Importing local modules
from data_processing import load_data, split_vectorize_fit_text
from save_model_report import save_models, save_report
from model_training import train_classifier

# Loading the data
data = load_data(os.path.join("in", "fake_or_real_news.csv"))

# Splitting, vectorizing, and fitting the text data
X_train_feats, X_test_feats, y_train, y_test, vectorizer = split_vectorize_fit_text(data, "text", "label", 500)

# Training logistic regression classifier
logreg_classifier = train_classifier(X_train_feats, y_train, classifier_type='logreg')

# Training MLP classifier
mlp_classifier = train_classifier(X_train_feats, y_train, classifier_type='mlp')

# Making predictions using logistic regression classifier
y_pred_logreg = logreg_classifier.predict(X_test_feats)

# Making predictions using MLP classifier
y_pred_mlp = mlp_classifier.predict(X_test_feats)

# Saving models
save_models(logreg_classifier, vectorizer, os.path.join("models"))
save_models(mlp_classifier, vectorizer, os.path.join("models"))

# Saving classification reports
save_report(y_test, y_pred_logreg, os.path.join("out", "logreg_report.txt"))
save_report(y_test, y_pred_mlp, os.path.join("out", "mlp_report.txt"))