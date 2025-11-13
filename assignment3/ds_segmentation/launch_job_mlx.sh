#!/bin/sh -l
#SBATCH -J ds_segmentation      # Name of your job in SLURM
#SBATCH --account=p200776
#SBATCH --qos=short
#SBATCH -p gpu                  # Partition name
#SBATCH -N 2                    # Number of nodes
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --output=ds_segmentation.out
#SBATCH --error=ds_segmentation.err
#SBATCH -c 7                    # Cores assigned to each task
#SBATCH --gres=gpu:1            # GPUs per node
#SBATCH --time=0-1:00:00        # Time limit

# Load required modules
module load env/release/2024.1
module load scikit-learn/1.5.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0

# Activate virtual environment from scratch space
VENV_PATH="/project/scratch/p200981/u103056/ds_env"
source ${VENV_PATH}/bin/activate

# Redirect cache directories to scratch to avoid disk quota issues
export TRITON_CACHE_DIR="/project/scratch/p200981/u103056/.triton"
export TORCH_HOME="/project/scratch/p200981/u103056/.torch"
export HF_HOME="/project/scratch/p200981/u103056/.cache/huggingface"

# Set pip cache and tmp directories
export PIP_CACHE_DIR="/project/scratch/p200981/u103056/pip_cache" 
export TMPDIR="/project/scratch/p200981/u103056/tmp" 
mkdir -p $PIP_CACHE_DIR $TMPDIR

# Create cache directories if they don't exist
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

# Verify Python version and deepspeed
echo "Python and DeepSpeed info"
which python
python --version
which deepspeed
echo "=== Cache directories ==="
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_HOME=$HF_HOME"
echo "================================="

# Set up distributed training environment variables for DeepSpeed
# Get the master node hostname
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== Distributed Setup ==="
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID will be set by srun"
echo "=========================="

# Run DeepSpeed training using srun (no deepspeed launcher needed)
# srun will handle the distributed execution across nodes
srun --mem=0 --export=ALL python cifar_segmentation_ds.py --deepspeed --deepspeed_config ds_config.json
