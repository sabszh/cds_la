# Linguistic Feature Extraction using spaCy

This script extracts linguistic features: Part-of-Speech (POS) tags and named entities from text files using spaCy.

## Requirements

- Python > 3.10.12
- `pandas` library
- `spacy` library
- `en_core_web_md` model for spaCy (downloadable using `python -m spacy download en_core_web_md`)

## Usage

1. Clone the repository or download the script.

2. Install the required packages by running:
    ```
    pip install -r requirements.txt
    ```

3. Run the script by executing:
    ```
    python script.py
    ```

## Description

The script performs the following tasks:

- **Cleaning Text**: Removes HTML tags from the text.
- **Processing Text**: Extracts linguistic features from the text files, including the relative frequency of POS tags (Nouns, Verbs, Adjectives, Adverbs) and counts of unique named entities (Person, Location, Organization).
- **Running the Script**: Processes text files located in specified directories and outputs the results as CSV files.

## File Structure
The script assumes the following directory structure:

project_root/
│
├── in/
│   └── USEcorpus/
│       └── Subfolders/
│           └── Data/
│
├── out/
│
├── src/
│   ├── requirements.txt
│   └── script.py
│
└── README.md

## Output

Processed data is saved as CSV files in the `out` directory. 
