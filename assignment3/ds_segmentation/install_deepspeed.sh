#!/bin/bash -l
# Install DeepSpeed in a virtual environment
# Using scratch space to avoid home directory quota limits
# The -l flag ensures module command is available

# Load Python module from PyTorch - MUST match the modules in launch_job_mlx.sh
module load env/release/2024.1
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0

# Define paths
SCRATCH_DIR="/project/scratch/p200981/u103056"
VENV_DIR="${SCRATCH_DIR}/ds_env"

echo "Creating virtual environment in scratch space: ${VENV_DIR}"
python -m venv "${VENV_DIR}"

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing DeepSpeed and dependencies..."
pip install deepspeed
pip install torch torchvision
pip install numpy scikit-learn matplotlib

echo "Installation complete!"
echo "Virtual environment is located at: ${VENV_DIR}"
echo "To activate the environment, run: source ${VENV_DIR}/bin/activate"

