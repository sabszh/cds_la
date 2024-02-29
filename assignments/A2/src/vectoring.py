######
# Script for function that vectorizes text 
######

def vectorizer(text, label, max_f, test_size = 0.2, ngram_range = (1,2),lowercase = True, max_df = 0.95, min_df = 0.95):
    '''
    This is a function for splitting data and vectorising.
    '''
    # Labeling the data
    X = data[text]
    y = data[label]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=test_size,   
                                                        random_state=42) # random state for reproducibility

    vectorizer = TfidfVectorizer(ngram_range = ngram_range,     
                             lowercase =  lowercase,       
                             max_df = max_df,           
                             min_df = 0.05,           
                             max_features = max_f)      

    return vectorizer






