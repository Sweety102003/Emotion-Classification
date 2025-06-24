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

echo "=== Virtual Environment Setup Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To install dependencies (if you add new ones), run:"
echo "make install"
echo ""
echo "To start the notebook: make notebook"
echo "To start the web app: make app" 