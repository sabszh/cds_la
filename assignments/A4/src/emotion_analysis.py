"""
Assignment: 4 - Emotion analysis with pretrained language models
Course: Language Analytics
Author: Sabrina Zaki Hansen
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from codecarbon import EmissionsTracker

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
        ax.bar(emotion_counts.index, emotion_counts.values, color=[emotion_colors.get(emotion) for emotion in emotion_counts.index])
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
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    emotion_colors = {
        'anger': 'red',
        'disgust': 'green',
        'fear': 'purple',
        'joy': 'yellow',
        'neutral': 'gray',
        'sadness': 'blue',
        'surprise': 'orange'
    }

    for i, emotion_label in enumerate(data['Emotion_Label'].unique()):
        emotion_data = data[data['Emotion_Label'] == emotion_label]
        counts = emotion_data['Season'].value_counts().sort_index()
        total_count = counts.sum()
        relative_freq = counts / total_count  
        ax = relative_freq.plot(kind='bar', ax=axs[i // 3, i % 3], color=emotion_colors.get(emotion_label, 'black'))
        ax.set_title("Relative frequency of " + r"$\bf{" + emotion_label + "}$" + " across all seasons")
        ax.set_xlabel('Seasons')
        ax.set_ylabel('Relative Frequency')

    plt.tight_layout()
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

def main():
    """
    Main function to execute the emotion analysis pipeline.

    Args:
        input_file (str): Path to the input data file.

    Returns:
        None
    """

    if not os.path.exists(os.path.join("out")):
        os.makedirs(os.path.join("out"))
    output_dir = os.path.join("out")
    input_file = os.path.join("in","Game_of_Thrones_Script.csv")

    # Start CodeCarbon tracker
    tracker = EmissionsTracker(project_name="Emotion Analysis", 
                              experiment_id="emotion_analysis",
                              output_dir = output_dir)

    # Check if predicted_emotions.csv exists
    predicted_emotions_file = os.path.join(output_dir, 'predicted_emotions.csv')
    if os.path.exists(predicted_emotions_file):
        # If the file exists, load the data from it
        tracker.start_task("load_data_existing_file")
        data = pd.read_csv(predicted_emotions_file)
        tracker.stop_task()
    else:
        # If the file doesn't exist, load data from input file and predict emotion scores
        tracker.start_task("load_data")
        data = pd.read_csv(input_file)
        tracker.stop_task()

        # Predict emotions
        tracker.start_task("predict_emotions")
        data = predict_emotion_scores(data)
        tracker.stop_task()

        # Save predicted emotions to a CSV file
        tracker.start_task("save_file_csv")
        save_predicted_emotions(data, output_dir)
        tracker.stop_task()

    # Plot distribution of emotion labels in each season and save plots
    tracker.start_task("plot_emotions_distribution")
    plot_season_emotions(data, output_dir)
    tracker.stop_task()

    # Plot relative frequency of each emotion label across all seasons and save plot
    tracker.start_task("plot_relative_frequency")
    plot_relative_emotion_freq(data, output_dir)
    tracker.stop_task()

    _ = tracker.stop()

if __name__ == "__main__":
    main()