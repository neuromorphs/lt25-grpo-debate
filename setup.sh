#!/bin/bash

# GRPO Training Setup Script
# This script installs the required dependencies for the GRPO training script

set -e

echo "Setting up GRPO training environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is required but not installed."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "Installing dependencies with $PIP_CMD..."

# Install main dependencies
$PIP_CMD install -r requirements.txt

echo "Setup completed successfully!"
echo ""
echo "To run the training script:"
echo "  python3 main.py"
echo ""
echo "Make sure you have sufficient GPU memory (recommended: 8GB+ VRAM)"
echo "The script will:"
echo "  1. Load the Qwen 2.5 3B Instruct model"
echo "  2. Set up PEFT with LoRA"
echo "  3. Load and prepare the GSM8K dataset"
echo "  4. Train using GRPO with multiple reward functions"
echo "  5. Test the model before and after training"
echo "  6. Save the trained LoRA adapters"