######
# Script for function that vectorizes text 
######

def save_models(classifier, vectorizer,output_path):
    dump(classifier, f"{filepath}LR_{classifier}.joblib")
    dump(vectorizer, f"{filepath}TFIDF_{vectorizer}.joblib")

def save_report(y_test,y_pred,output_path):
    class_report = metrics.classification_report(y_test, y_pred)
    
    with open(output_path, "w") as report_file:
    report_file.write(class_report)
    print("Report file saved")