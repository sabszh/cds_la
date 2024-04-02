# Query Expansion with Word Embeddings

This script performs query expansion with word embeddings for song lyrics analysis. It finds similar words to a given search term using a pre-trained word embedding model and then calculates the percentage of songs by a specified artist containing terms from the expanded query.

## Usage

To use this script, follow these steps:

1. Ensure you have Python installed on your system.

2. Install the required packages using the following command:
    ```
    pip install -r requirements.txt
    ```

3. Execute the script with the following command:
    ```
    python script.py <file_path> <artist> <search_term>
    ```

    - `<file_path>`: Path to the file containing song lyrics.
    - `<artist>`: Name of the artist to analyze.
    - `<search_term>`: Word to expand the query.

## Output

The script outputs the percentage of songs by the specified artist containing words related to the search term and similar terms. It also saves the results to a CSV file named `results.csv` in the `out` directory.

## Example

Suppose you want to analyze songs by the artist "Taylor Swift" with the search term "love". You would execute the script as follows:
```
python script.py songs.txt "Taylor Swift" "love"
```

This would analyze the songs in `<file_path>`, expand the query with similar words to "love", calculate the percentage of Taylor Swift's songs featuring terms related to "love", and save the results to `out/results.csv`.

```
Output: 
48.27% of Taylor Swift's songs contain words related to love and similar terms.
```

## File Structure

The project assumes the following directory structure:

```
.
project_root/
│
├── in/
│ └── songs_lyrics.csv
│
├── out/
│ └── results.csv
│
├── src/
│ └── lyrics_analysis.py
│
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.x
- pandas
- numpy
- gensim