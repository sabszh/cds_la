# Assignment 3: Query Expansion with Word Embeddings
This script performs query expansion with word embeddings for song lyrics analysis. It finds similar words to a given search term using a pre-trained word embedding model and then calculates the percentage of songs by a specified artist containing terms from the expanded query.

## Data Source
The song lyrics used for this analysis was obtained from [this dataset](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). The data contains 4 columns: `artist`, `song`, `link`, `text`.

## Requirements
- Python > 3.10.12
- `gensim` library
- `numpy` library
- `pandas` library
- `codecarbon` library

## Usage
To use this script, follow these steps:
1. Clone or download the repository.

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh <artist> <search_term>
    ```
    - `<"artist">`: Name of the artist to analyze (important: write in quotation)
    - `<"search_term">`: Word to expand the query (important: write in quotation)

## Example
Suppose you want to analyze songs by the artist "Taylor Swift" with the search term "love". You would execute the script as follows:
```
bash run.sh "Taylor Swift" "love"
```

This would analyze the songs in the file, expand the query with similar words to "love", calculate the percentage of Taylor Swift's songs featuring terms related to "love", and save the results to `out/results.csv`.

*Example of output*
```
Output: 
48.27% of Taylor Swift's songs contain words related to love and similar terms.
```

## Script Overview
This Python script is developed for conducting query expansion with word embeddings, primarily for analyzing song lyrics. It uses word embedding models to expand search terms and then assesses the prevalence of those terms in songs by a specified artist. Below is a concise summary of the script's functionalities:

### Command-line Arguments
The script expects the following command-line arguments:
- `artist`: Name of the artist whose songs will be analyzed.
- `search_term`: The word used as the base for query expansion.

### Functions
1. **`parse_arguments()`**: Parses the command-line arguments provided to the script.
2. **`load_lyrics(file_path)`**: Loads song lyrics from a file.
3. **`find_similar_words(model, word, top_n=10)`**: Retrieves similar words to a given word using a word embedding model.
4. **`calculate_percentage(artist_songs, expanded_query)`**: Calculates the percentage of songs by the artist featuring terms from the expanded query.
5. **`save_to_csv(output_path, artist, search_term, percentage)`**: Saves analysis results to a CSV file.

### Carbon Emission Tracking Integration

The script integrates the CodeCarbon library to monitor and track carbon emissions during the analysis process. It includes the following functionality:

- Initializes the tracker with project details.
- Starts tracking emissions for specific tasks (e.g., data loading, model loading).
- Stops tracking emissions after completing each task.
- Saves emission data to the specified output directory for further analysis.

### Main Functionality

The main function of the script orchestrates the entire process:

- Parses command-line arguments to determine the artist and search term.
- Loads song lyrics from a provided file.
- Loads a pre-trained word embedding model.
- Expands the query term with similar words.
- Filters songs by the specified artist.
- Calculates the percentage of songs containing the expanded query terms.
- Saves analysis results to a CSV file.
- Prints the percentage of relevant songs based on the query expansion.

## Output Summary
The script outputs the percentage of songs by the specified artist containing words related to the search term and similar terms. It also saves the results to a CSV file named `results.csv` in the `out` directory.

## Discussion of Limitations and Possible Steps to Improvement
While the current implementation of the script serves its purpose for query expansion and analysis of song lyrics, there are several limitations and areas for potential improvement.

Firstly, the choice of word embedding model, such as GloVe, while common and widely used, may not always capture the nuanced relationships between words effectively. Exploring different word embedding models or training custom embeddings on domain-specific data could potentially enhance the relevance of the expanded terms. Moreover, the absence of a benchmarking or evaluation tool makes it challenging to gauge the performance of the script and/or model accurately.

Additionally, the query expansion strategy employed by the script is relatively simplistic, relying on finding similar words to the given search term. While this approach is effective in many cases, it may overlook context-specific relationships between words. Incorporating more sophisticated query expansion techniques, such as semantic similarity or context-aware embeddings, could lead to more nuanced and relevant expansions. Furthermore, the script does not explicitly address the challenges posed by ambiguous or polysemous words. As a result, the expanded query may include irrelevant or ambiguous terms that could lead to inaccurate analysis results. Incorporating techniques to disambiguate words or considering contextual information could mitigate this issue.

## CodeCarbon Tracking
To track emissions, the script utilizes CodeCarbon. Emission data for each task is recorded in a CSV files located in the `out` directory.

For a more detailed analysis of these results, please see Assignment 5.

## File Structure
The project assumes the following directory structure:

```
.
A3/
│
├── in/
│ └── Spotify Million Song Dataset_exported.csv
│
├── out/
│ ├── emissions_base_{UUID}.csv
│ ├── emissions.csv # This file should just be ignored
│ └── results.csv
│
├── src/
│ └── lyrics_analysis.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```