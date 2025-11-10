#!/bin/sh -l
#SBATCH -J ds_segmentation      # Name of your job in SLURM
#SBATCH -N 2                    # Number of nodes
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --output=ds_segmentation.out
#SBATCH --error=ds_segmentation.err
#SBATCH -c 7                    # Cores assigned to each task
#SBATCH --gres=gpu:1            # GPUs per node
#SBATCH --time=0-4:00:00        # Time limit
#SBATCH -p gpu                  # Partition

# Load required modules
module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Activate virtual environment if you have one
# source ds_env/bin/activate

# Set CUDA environment variables
export CUDA_HOME=/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Debug: confirm CUDA is available on all nodes
echo "=== Node and CUDA info ==="
srun bash -c 'echo "--- $(hostname) ---"; 
              echo "CUDA_HOME=$CUDA_HOME"; 
              which nvcc || echo "nvcc not found"; 
              python -c "import torch; print(\"torch.cuda.is_available=\", torch.cuda.is_available(), \"n_gpus=\", torch.cuda.device_count())"'
echo "==========================="

# Run DeepSpeed training
srun deepspeed cifar_segmentation_ds.py --deepspeed --deepspeed_config ds_config.json
