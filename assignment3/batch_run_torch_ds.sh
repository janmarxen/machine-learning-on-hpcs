#!/bin/sh -l
#SBATCH -J ds_cifar            # Name of your job in SLURM
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --output=ds_cifar.txt        # name of the file to save the output
#SBATCH -c 7                                # Cores assigned to each tasks
#SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00
#SBATCH -p gpu

#### The following commented code ####
#### Is in case you want to try   ####
#### using a hosts file           ####
#hosts_file="hosts.txt"
#scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file
#
## Collect public key and accept them
#while read -r node; do
#    ssh-keyscan "$node" >> ~/.ssh/known_hosts
#done < "$hosts_file"
#
## Create the host file containing node names and the number of GPUs
#function makehostfile() {
#perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
#$slots=4 if $slots==0;
#@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
#print map { "$b$_ slots=$slots\n" } @nodes'
#}
#makehostfile > hostfile

module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Check if matches the name of your virtual environment (if cyou are using one)
source ds_env/bin/activate

export CUDA_HOME=/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export DS_BUILD_OPS=0
#export DS_BUILD_AIO=0

# === Debug: confirm both nodes see CUDA ===
echo "=== Node and CUDA info ==="
srun bash -c 'echo "--- $(hostname) ---"; 
              echo "CUDA_HOME=$CUDA_HOME"; 
              which nvcc || echo "nvcc not found"; 
              python -c "import torch; print(\"torch.cuda.is_available=\", torch.cuda.is_available(), \"n_gpus=\", torch.cuda.device_count())"'
echo "==========================="

# === Run DeepSpeed (no hostfile needed) ===
srun deepspeed torch_cnn_ds.py --deepspeed --deepspeed_config ds_config.json

# Alternative with host_file
#srun deepspeed \
#     --num_nodes=$SLURM_JOB_NUM_NODES \
#     --num_gpus=1 \
#     --hostfile hostfile \
#     torch_cnn_ds.py \
#     --deepspeed \
#     --deepspeed_config ds_config.json
