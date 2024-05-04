"""
Assignment: 4 - Emotion analysis with pretrained language models
Course: Language Analytics
Author: Sabrina Zaki Hansen
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

######
# Defining functions
######

# Predicting emotion scores from pipeline classifier
def predict_emotion_scores(data):
    """
    Predict emotion scores for all lines in the data using a pretrained language model.

    Args:
        data (pd.DataFrame): DataFrame containing the script data.

    Returns:
        pd.DataFrame: DataFrame with added emotion scores.
    """
    classifier = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base")
    
    # Get sentences from the DataFrame
    sentences = [str(sentence) for sentence in data['Sentence'].tolist()]
    
    # Predict emotion scores for all sentences
    emotion_scores = classifier(sentences)

    # Extract emotion labels and scores
    labels = [entry['label'] for entry in emotion_scores]
    data['Emotion_Label'] = labels

    return data

# Plotting predicted emotion
def plot_season_emotions(data, output_dir):
    """
    Plot the distribution of all emotion labels in each season and save the plots.

    Args:
        data (pd.DataFrame): DataFrame containing the script data with emotion scores.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    # Define colors for each emotion
    emotion_colors = {
        'anger': 'red',
        'disgust': 'green',
        'fear': 'purple',
        'joy': 'yellow',
        'neutral': 'gray',
        'sadness': 'blue',
        'surprise': 'orange'
    }

    seasons = data['Season'].unique()
    num_seasons = len(seasons)

    # Create subplots based on the number of seasons, arranging them in a grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 20))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through each season and corresponding subplot
    for i, (season, ax) in enumerate(zip(seasons, axes)):
        # Filter data for the current season
        season_data = data[data['Season'] == season]
        # Count the occurrences of each emotion label
        emotion_counts = season_data['Emotion_Label'].value_counts()
        # Plot the bar chart for emotion distribution in the current season
        ax.bar(emotion_counts.index, emotion_counts.values, color=[emotion_colors.get(emotion, 'black') for emotion in emotion_counts.index])
        ax.set_title(f'Distribution of Emotion Labels in {season}')
        ax.set_xlabel('Emotion Label')
        ax.set_ylabel('Frequency')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_seasons_emotions.png'))
    plt.close()

# Relative frequency
def plot_relative_emotion_freq(data, output_dir):
    """
    Plot the relative frequency of each emotion label across all seasons and save the plot.

    Args:
        data (pd.DataFrame): DataFrame containing the script data with emotion scores.
        output_dir (str): Directory to save the plot.

    Returns:
        None
    """
    seasons = data['Season'].unique()
    plt.figure()
    width = 0.8 / len(seasons) 
    offset = -0.4  
    for season in seasons:
        season_data = data[data['Season'] == season]
        relative_freq = season_data['Emotion_Label'].value_counts(normalize=True)
        labels = relative_freq.index
        values = relative_freq.values
        plt.bar(labels, values, width=width, align='center', label=f'{season}', alpha=0.8)
        offset += width  
    plt.title('Relative Frequency of Emotion Labels Across Seasons')
    plt.xlabel('Emotion Label')
    plt.ylabel('Relative Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'relative_emotion_frequency.png'))
    plt.close()

# Save predicted emotions to a CSV file
def save_predicted_emotions(data, output_dir):
    """
    Save the DataFrame with predicted emotions to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame containing the script data with predicted emotions.
        output_dir (str): Directory to save the CSV file.

    Returns:
        None
    """
    data.to_csv(os.path.join(output_dir, 'predicted_emotions.csv'), index=False)

###### 
# Main function
######

def main(input_file):
    """
    Main function to execute the emotion analysis pipeline.

    Args:
        input_file (str): Path to the input data file.

    Returns:
        None
    """
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)  # Create "out" directory if it doesn't exist

    # Check if predicted_emotions.csv exists
    predicted_emotions_file = os.path.join(output_dir, 'predicted_emotions.csv')
    if os.path.exists(predicted_emotions_file):
        # If the file exists, load the data from it
        data = pd.read_csv(predicted_emotions_file)
    else:
        # If the file doesn't exist, load data from input file and predict emotion scores
        data = pd.read_csv(input_file)
        data = predict_emotion_scores(data) 
        # Save predicted emotions to a CSV file
        save_predicted_emotions(data, output_dir)

    # Plot distribution of emotion labels in each season and save plots
    plot_season_emotions(data, output_dir)

    # Plot relative frequency of each emotion label across all seasons and save plot
    plot_relative_emotion_freq(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion analysis of Game of Thrones script data.")
    parser.add_argument("input_file", type=str, help="Path to the input data file")
    args = parser.parse_args()

    main(args.input_file)