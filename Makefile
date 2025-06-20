# Makefile for the Emotion Classification Project

# Define the Python interpreter from the virtual environment
PYTHON = venv/bin/python
PIP = venv/bin/pip

# Phony targets are not actual files
.PHONY: all install notebook app clean help

# Default target
all: help

# Install dependencies
install:
	@echo "Installing dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt
	@echo "Installation complete."

# Run Jupyter Notebook
notebook:
	@echo "Starting Jupyter Notebook server..."
	$(PYTHON) -m notebook --notebook-dir=notebook

# Run the Streamlit web application
app:
	@echo "Starting Streamlit web app..."
	venv/bin/streamlit run streamlit_app/app.py

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."

# Display help
help:
	@echo "Available commands:"
	@echo "  make install   - Install project dependencies"
	@echo "  make notebook  - Start the Jupyter Notebook server"
	@echo "  make app       - Run the Streamlit web application"
	@echo "  make clean     - Remove temporary Python files" 