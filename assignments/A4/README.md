# Assignment 4 - Emotion analysis with pretrained language models
## Overview
This script performs emotion analysis on the scripts of the television show *Game of Thrones*. It predicts emotion scores for each line in the scripts using a pretrained language model, and then analyzes the distribution of emotions across seasons as well as the relative frequency of each emotion label across all seasons.

## How to Use
1. **Input Data**: Provide the path to the input data file containing the Game of Thrones scripts.
2. **Output**: The script generates visualizations showing the distribution of emotion labels in each season and the relative frequency of each emotion label across all seasons. These visualizations are saved in the `out` directory.
3. **Execution**: Run the script with the specified input file path as a command line argument.

## Prerequisites
- Python 3
- Required Libraries: `pandas`, `matplotlib`, `transformers`, `torch`

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Command Line Usage
Run the script with the following command:
```
python script.py input_file_path
```
Replace `script.py` with the name of the script file and `input_file_path` with the path to the input data file.

## Example
```
python emotion_analysis.py in/game_of_thrones_scripts.csv
```

## Output
The script generates two types of visualizations:
1. **Distribution of Emotion Labels by Season**: Bar charts showing the distribution of emotion labels in each season.
   - These visualizations are saved as `all_seasons_emotions.png`.
2. **Relative Frequency of Emotion Labels Across Seasons**: Bar chart showing the relative frequency of each emotion label across all seasons.
   - This visualization is saved as `relative_emotion_frequency.png`.

## Additional Notes
- The pretrained language model used for emotion analysis is `j-hartmann/emotion-english-distilroberta-base`.
- The output directory `out` is automatically created if it doesn't exist, and all generated visualizations are saved there.

## Repository Structure
Ensure that your repository follows the structure outlined below:
```
root-directory/
│
├── emotion_analysis.py
├── requirements.txt
├── in/
│   └── game_of_thrones_scripts.csv
├── out/
│   └── (output visualizations)
└── README.md
```

## Summary and interpretation of plots