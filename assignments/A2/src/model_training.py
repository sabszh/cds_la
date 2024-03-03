######
# Script for model training
######

# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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