#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Instal scipy
pip install scipy==1.11.0

# Install requirements
pip install -r requirements.txt

# Deactivate virtual environment
deactivate
