#!/usr/bin/python

"""
Assignment 3 - Query expansion with word embeddings
"""

# Importing packages
import sys
import os
import argparse
import pandas as pd
import numpy as np
import gensim.downloader as api

sys.path.append(os.path.join('..'))

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Query expansion with word embeddings for song lyrics analysis")
    parser.add_argument("file_path", type=str, help="Path to the file containing song lyrics")
    parser.add_argument("artist", type=str, help="Name of the artist to analyze")
    parser.add_argument("search_term", type=str, help="Word to expand the query")
    args = parser.parse_args()
    
    return parser.parse_args()

def load_lyrics(file_path):
    """
    Load song lyrics from a file.

    Args:
        file_path (str): Path to the file containing song lyrics.

    Returns:
        list: List of song lyrics.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lyrics = file.read().splitlines()
    return lyrics

def find_similar_words(model, word, top_n = 10):
    """
    Find similar words to a given word using a word embedding model.

    Args:
        model: Word embedding model.
        word (str): Word to find similar words for.
        top_n (int): Number of similar words to retrieve.

    Returns:
        list: List of similar words.
    """
    similar_words = model.most_similar(word, topn=top_n)
    return [word for word, _ in similar_words]

def calculate_percentage(artist_songs, expanded_query):
    """
    Calculate the percentage of songs by the artist featuring terms from the expanded query.

    Args:
        artist_songs (list): List of songs by the artist.
        expanded_query (list): Expanded query terms.

    Returns:
        float: Percentage of songs featuring terms from the expanded query.
    """
    total_songs = len(artist_songs)
    songs_with_query_terms = sum(1 for song in artist_songs if any(term in song.lower() for term in expanded_query))
    return (songs_with_query_terms / total_songs) * 100 if total_songs > 0 else 0

def save_to_csv(output_path, artist, search_term, percentage):
    """
    Save analysis results to a CSV file.

    Args:
        output_path (str): Path to the output directory.
        artist (str): Name of the artist.
        search_term (str): Search term used for analysis.
        percentage (float): Percentage of songs featuring terms from the expanded query.
    """
    output_file = os.path.join(output_path, 'results.csv')
    percentage = round(percentage, 2)
    df = pd.DataFrame({'Artist': [artist], 'Search Term': [search_term], 'Percentage': [percentage]})
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

def main():
    """
    Main function to execute the query expansion and analysis process.
    """
    # Parse the arguments
    args = parse_arguments()

    # Load song lyric
    lyrics = load_lyrics(args.file_path)

    # Load word embedding model
    model = api.load("glove-wiki-gigaword-50")

    # Find similar words to the search term
    similar_words = find_similar_words(model, args.search_term)

    # Expand query with similar words
    expanded_query = [args.search_term] + similar_words
    
    # Filter artists's songs
    artist_songs = [song for song in lyrics if args.artist.lower() in song.lower()]

    # Calculate percentage of songs featuring terms from the expanded query
    percentage = calculate_percentage(artist_songs, expanded_query)

    # Save results to CSV
    save_to_csv('out', args.artist, args.search_term, percentage)

    # Print results
    print(f"{percentage:.2f}% of {args.artist}'s songs contain words related to {args.search_term} and similar terms.")
    
if __name__=="__main__":
    main()