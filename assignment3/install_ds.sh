module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0


python -m venv ds_env --system-site-packages

# Activate the environment
echo "2. Activating environment..."
source ds_env/bin/activate

# Upgrade pip
echo "3. Upgrading pip..."
pip install --upgrade pip

pip install deepspeed

ds_report