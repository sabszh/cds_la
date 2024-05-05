# Assignment 1: Extracting linguistic features using spaCy
This script extracts linguistic features from the USEcorpus. It extracts Part-of-Speech (POS) tags and named entities using spaCy and saves the output as CSV files for each subfolder in the chosen dataset. It also tracks carbon emission.

## Data Source
The corpus used for this analysis is: *The Uppsala Student English Corpus (USE)*. You can access more documentation via [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

USE was set up with the aim of creating a powerful tool for research into the process and results of foreign language teaching and acquisition, as manifest in the written English of Swedish university students.

### Contents
The corpus consists of 1,489 essays written by 440 Swedish university students of English at three different levels. The essays cover set topics of different types.

**First-term essays:**
- Evaluation (a1): Students describe their experience of the English language, evaluating their reading, writing, speaking, and listening proficiency. Personal, involved style.
- Argumentation (a2): Students argue for or against a statement concerning a topical issue. Formal style.
Reflections (a3): Students reflect on the medium of television and its impact on people, or on related issues of their choice. Personal/formal style.
- Literature course assignment (a4): Students choose between a discussion of theme/character/narrator and a close-reading based analysis of a set passage. Formal style.
- Culture course assignment (a5): Students study topics in set secondary sources and compose an essay using this material, often quoting and listing these sources.

**Second-term essays:**
- Causal analysis (b1): Students discuss causes of some recent trend of their choice. Formal style.
- Argumentation (b2): Students present counter-arguments to views expressed in articles or letters to the editor. Similar in approach and tone to essay a1.
- Short papers in English linguistics (b3): Academic style.
- English literature (b4): Discussion of character, theme etc., produced in a survey course, dealing with Shakespeare’s Julius Ceasar or contemporary novels.
- American literature (b5): Similar to b4.

**Third-term essays:**
- Literature course essays (c1): Longer essays, all literature course assignments.

## Requirements
- Python > 3.10.12
- `pandas` library
- `spacy` library
- `codecarbon` library
- `en_core_web_md` model for spaCy (downloadable using `python -m spacy download en_core_web_md`)

## Usage
1. Clone or download the repository.

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh
    ```

## Script overview
This Python script, designed for extracting linguistic features using spaCy, facilitates text processing and analysis. IBelow is a breakdown of its functionalities:

### Functions
1. **`cleaning_text(text)`**: Cleans the input text by removing HTML tags.
2. **`processing_text(file_path)`**: Processes the text file located at the given file path and extracts linguistic features, including the relative frequency of POS tags and counts of unique entities.
3. **`main()`**: The main function orchestrates the execution of the script. It iterates over directories containing text files, processes each file to extract linguistic features, and saves the results to CSV files.

### Carbon Emission Tracking Integration

The script integrates the Carbon Emission Tracker library to monitor and track carbon emissions during the execution of linguistic feature extraction tasks. It includes the following functionality:

- Initializes the tracker with project details for each assignment folder.
- Starts tracking emissions for each assignment folder.
- Stops tracking emissions after completing processing for each assignment folder.
- Saves emission data to a CSV file for further analysis.

### Main Functionality
The main function of the script performs the following steps:
1. Creates an output directory if it does not exist.
2. Processes text files located in subfolders of the input directory (`in/USEcorpus`).
3. Extracts linguistic features from each text file and saves the results to CSV files.
4. Prints progress messages indicating the completion of processing for each assignment folder.

## Output Summary
Processed data is saved as CSV files in the `out` directory. Each file is for each folder  It contains the followiing columns:
- Filename: The name of the text file being processed.
- Relative Frequency of NOUN: The relative frequency of nouns in the text.
- Relative Frequency of VERB: The relative frequency of verbs in the text.
- Relative Frequency of ADJ: The relative frequency of adjectives in the text.
- Relative Frequency of ADV: The relative frequency of adverbs in the text.
- Unique PER: The count of unique person entities detected in the text.
- Unique LOC: The count of unique location entities detected in the text.
- Unique ORG: The count of unique organization entities detected in the text.

Overall the outputs do not give much insight, but the results are ready to be shared or used for future analysis.
Moreover, the folder also contains output from CodeCarbon emission tracking `emissions.csv`.

## Discussion of Limitations and Possible Steps to Improvement
The linguistic patterns and characteristics present in the USE corpus may differ significantly from a general English language corpora. Thus, the generic nature of the linguistic feature extraction script may not fully capture the unique linguistic nuances and challenges exhibited by data of Swedish learner of English. Esspecially as the the data is from different times and different styles. Simply plotting the different features for each essay type could bring insight to linguistic differences, and by further qualitative expection bring clarity to whether it is due to model ineffecenicy of linguistic differences across the essays. 

Moreover, this script relies on spaCy and its English model (`en_core_web_md`), impacting linguistic feature extraction accuracy. Entity recognition precision is affected by text quality and model relevance, potentially leading to affecting the performance.

Improvements could involve adopting a larger or domain-specific model to improve accuracy, though this would still be with possible trade-offs. Optimizing code efficiency for processing large text volumes, through parallel processing or algorithmic improvements, is also beneficial. 

## CodeCarbon Tracking
To track emissions, the script utilizes CodeCarbon. Emission data for each task is recorded in a CSV file named `emissions.csv` located in the `out` directory.

For a more detailed analysis of these results, please see Assignment 5.

## File Structure
The script assumes the following directory structure:

```
.
A1/
│
├── in/
│   └── USEcorpus/
│       ├── a1/
│       │   ├── 0100.a1.txt
│       │   └── ...
│       ├── a2/
│       └── ...    
│
├── out/
│   ├── a1_linguistic_features.csv
│   ├── a2_linguistic_features.csv
│   ├── ...
│   └── emissions.csv
│
├── src/
│   └── extract_features.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```