######
# Model training
######

def train_classifier(X_train, y_train, classifier_type='logistic_regression', **kwargs):
    if classifier_type == 'logistic_regression':
        classifier = LogisticRegression(**kwargs)
    elif classifier_type == 'neural_network':
        classifier = MLPClassifier(**kwargs)
    else:
        raise ValueError("Invalid classifier type. Supported types are 'logistic_regression' and 'neural_network'.")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test_feats)
    return classifier, y_pred

classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 971)

classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)