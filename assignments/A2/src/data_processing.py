######
# Script for function that vectorizes text 
######

def load_data(filename):
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
        Tuple: X_train, X_test, y_train, y_test
    '''
    # Extracting text and labels from the data
    X = data[text_column]
    y = data[label_column]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)

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





