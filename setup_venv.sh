#!/bin/bash

echo "=== Setting up Virtual Environment for Emotion Classification Project ==="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Install additional packages that might be missing
echo "Installing additional packages..."
pip install joblib

echo "=== Virtual Environment Setup Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the emotion classification pipeline, run:"
echo "python scripts/emotion_classification_pipeline.py" 