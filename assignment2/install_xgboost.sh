#!/bin/bash

# Script to create Python environment and install XGBoost
# Save this as setup_xgboost.sh and run: bash setup_xgboost.sh

echo "=== Setting up Python environment with XGBoost ==="

# Load modules
module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b

# Create Python virtual environment
echo "1. Creating Python virtual environment..."
python3 -m venv xgboost_env

# Activate the environment
echo "2. Activating environment..."
source xgboost_env/bin/activate

# Upgrade pip
echo "3. Upgrading pip..."
pip install --upgrade pip

# Install XGBoost
echo "4. Installing XGBoost..."
pip install xgboost
#pip install pandas

# Verify installation
echo "6. Verifying installation..."
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"

echo "=== Setup complete! ==="
echo "To activate the environment: source xgboost_env/bin/activate"
echo "To deactivate: deactivate"