#!/bin/bash

#using virtual environment name 'meng'
VENV_DIR="meng"

python3 -m venv $VENV_DIR

# check if created successfully
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment created successfully."

    # activate venv
    source $VENV_DIR/bin/activate
    echo "Activated venv $VENV_DIR successfully."

    # check for requirements.txt file
    if [ -f "requirements.txt" ]; then
        # Install the required packages
        pip install -r requirements.txt
        echo "Installing from requirements.txt"
    else
        echo "requirements.txt file not found."
    fi
else
    echo "Failed to create virtual environment."
fi