#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the script
python src/lyrics_analysis.py "$@"

# Deactivate the virtual environment
deactivate
