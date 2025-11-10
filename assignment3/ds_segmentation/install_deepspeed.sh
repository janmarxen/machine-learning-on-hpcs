#!/bin/bash
# Install DeepSpeed in a virtual environment

echo "Creating virtual environment..."
python -m venv ds_env

echo "Activating virtual environment..."
source ds_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing DeepSpeed and dependencies..."
pip install deepspeed
pip install torch torchvision
pip install numpy scikit-learn matplotlib

echo "Installation complete!"
echo "To activate the environment, run: source ds_env/bin/activate"
