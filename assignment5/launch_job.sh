#!/bin/bash
#SBATCH --job-name=assignment5
#SBATCH --output=assignment5.out
#SBATCH --error=assignment5.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load required modules (customize as needed)
# module load python/3.9

# Activate virtual environment if needed
# source venv/bin/activate

# Run your code here
echo "Running assignment5..."
# python main.py
