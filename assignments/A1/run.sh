#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the script
python src/extract_features.py

# Deactivate the virtual environment
deactivate
