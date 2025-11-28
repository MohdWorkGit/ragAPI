#!/bin/bash

###############################################################################
# Quick Install Script - Minimal interaction
# For automated/CI environments
###############################################################################

set -e

echo "Installing Python dependencies..."
pip3 install -r requirements.txt --user --quiet

echo "Downloading spaCy model..."
python3 -m spacy download en_core_web_sm --quiet

echo "Creating directories..."
mkdir -p writers_db docs

echo "✓ Installation complete!"
echo "Run 'python3 test_enhanced_extraction.py' to test the installation"
