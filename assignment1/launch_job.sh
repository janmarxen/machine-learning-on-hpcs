#!/bin/bash -l 
#SBATCH --job-name=assignment1
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --account=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=assignment1.out
#SBATCH --error=assignment1.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Load necessary modules for this assignment
module load data/scikit-learn/1.4.0-gfbf-2023b
module load vis/matplotlib/3.8.2
module load lang/SciPy-bundle/2023.11-gfbf-2023b
module load bio/Seaborn/0.13.2-gfbf-2023b

# Print loaded modules
module list
echo ""

# Run Python unbuffered so stdout is flushed immediately
PYTHONUNBUFFERED=1 srun python3 benchmark_linear_regression.py

echo ""
echo "Job completed at: $(date)"

