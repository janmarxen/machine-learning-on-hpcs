#!/bin/bash -l
#SBATCH --job-name=assignment2
#SBATCH --output=assignment2.out
#SBATCH --error=assignment2.err
#SBATCH --time=01:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16

# TODO: Explain the previous job specification clearly!
# Load required modules 
module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load mpi/OpenMPI
module load lib/mpi4py/3.1.5-gompi-2023b

# Install xgboost in virtual environment
./install_xgboost.sh

# Activate virtual environment 
source xgboost_env/bin/activate

# Run code 
# Naive contiguous assignment (original behavior)
# srun python xgboost_single_hyper.py --distribution contiguous
# Round-robin (default, best for typical cases)
srun python xgboost_single_hyper.py --distribution roundrobin
# Shuffle with seed
# srun python xgboost_single_hyper.py --distribution shuffle --seed 42
