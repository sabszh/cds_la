#!/usr/bin/bash

# create virtual environment
python -m venv env

# activate virtual environment
source ./env/bin/activate

# install requirements
pip install --upgrade pip
pipreqs src --savepath requirements.txt

# close the virtual environment
deactivate