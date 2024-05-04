#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the script
python src/text_classification.py "$@"

# Deactivate the virtual environment
deactivate
