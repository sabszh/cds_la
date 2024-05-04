#!/usr/bin/python

"""
Extracting linguistic features using spaCy
"""

# Loading in packages (run requirements.txt to install packages)
import os
import pandas as pd
import spacy
import re
import argparse

# Loading spacy
nlp = spacy.load("en_core_web_md")

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train classifiers on data.")
    parser.add_argument('classifier_type', type=str, choices=['logreg', 'mlp'], default='logreg', help="Type of classifier to train (logreg or mlp)")
    args = parser.parse_args()
    return args

def cleaning_text(text):
    """
    Clean the input text by removing HTML tags.
    
    Args:
    text (str): The input text with HTML tags.
    
    Returns:
    str: The cleaned text without HTML tags.
    """
    return re.sub(r'<.*?>', '', text)

def processing_text(file_path):
    """
    Process the text file located at the given file path and extract linguistic features.
    
    Args:
    file_path (str): The path to the text file.
    
    Returns:
    dict: A dictionary containing relative frequency of POS tags and counts of unique entities.
    """
    with open(file_path, "r", encoding="latin-1") as file:
        text = cleaning_text(file.read())
    
    doc = nlp(text)
    
    pos_tags = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    unique_entities = {"PER":0, "LOC":0, "ORG":0}
    
    for token in doc:
        if token.pos_ in pos_tags:
            pos_tags[token.pos_] += 1
    
    for ent in doc.ents:
        if ent.label_ in unique_entities:
            unique_entities[ent.label_] += 1
        
    total_pos = sum(pos_tags.values())
    relative_freq_pos = {key : value/total_pos for key, value in pos_tags.items()}
    
    data = {
        "Relative Frequency of NOUN": relative_freq_pos.get("NOUN", 0),
        "Relative Freqency of VERB": relative_freq_pos.get("VERB", 0),
        "Relative Freqency of ADJ": relative_freq_pos.get("ADJ", 0),
        "Relative Freqency of ADV": relative_freq_pos.get("ADV", 0),
        "Unique PER": unique_entities.get("PER", 0),
        "Unique LOC": unique_entities.get("LOC", 0),
        "Unique ORG": unique_entities.get("ORG", 0)
    }
    
    return data

def main():
    """
    Main function to execute the script.
    """
    data_path = os.path.join("in", "USEcorpus")
    output_path = os.path.join("out")
    dirs = sorted(os.listdir(data_path))

    for directory in dirs:
        subfolder = os.path.join(data_path, directory) 
        filenames = sorted(os.listdir(subfolder))
        results = []
    
        for text_file in filenames:
            file_path = os.path.join(subfolder, text_file)
            file_data = processing_text(file_path)
            results.append({"Filename": text_file, **file_data}) 
    
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_path, f"{directory}_linguistic_features.csv"), index=False)
        print(f"Processed {directory} folder")

if __name__ == "__main__":
    main()