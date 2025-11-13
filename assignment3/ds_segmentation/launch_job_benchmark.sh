#!/bin/sh -l
#SBATCH --account=p200776
#SBATCH --qos=short
#SBATCH -p gpu                  # Partition name
#SBATCH -c 7                    # Cores assigned to each task
#SBATCH --time=0-1:00:00        # Time limit
# Note: --nodes, --ntasks-per-node, --gres, --output, --error are set by benchmark_configs.sh

# Record start time
START_TIME=$(date +%s)

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

# Print configuration info
echo "========================================"
echo "DeepSpeed Training Benchmark"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Node list: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

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
echo ""
echo "Starting training..."
echo "========================================"

TRAIN_START=$(date +%s)
srun --mem=0 --export=ALL python cifar_segmentation_ds.py --deepspeed --deepspeed_config ds_config.json
TRAIN_END=$(date +%s)

# Calculate elapsed time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TRAINING_TIME=$((TRAIN_END - TRAIN_START))

echo "========================================"
echo "Benchmark Complete"
echo "========================================"
echo "End time: $(date)"
echo "Total job time: ${TOTAL_TIME} seconds ($((TOTAL_TIME / 60)) minutes)"
echo "Training time: ${TRAINING_TIME} seconds ($((TRAINING_TIME / 60)) minutes)"
echo "Setup overhead: $((TOTAL_TIME - TRAINING_TIME)) seconds"
echo "========================================"
