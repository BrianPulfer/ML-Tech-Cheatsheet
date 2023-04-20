#!/bin/bash

#SBATCH --job-name=template_job       # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=template@mail.com # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU (for more CPUs, use 'srun' --> OpenMPI)
#SBATCH --ntasks-per-node=1           # If using multiple CPUs per node
#SBATCH --nodes=1                     # Number of nodes to use
#SBATCH --cpus-per-task=1             # Single-threaded
#SBATCH --mem-per-cpu=1gb             # Memory per CPU request
#SBATCH --partition=gpu_partition     # Partition to be used
#SBATCH --gpus=4                      # Number of GPUs
#SBATCH --time=00:05:00               # Time limit days-hrs:min:sec
#SBATCH --output=template_out_%j.log     # Standard output log
#SBATCH --error=template_err_%j.log      # Standard error log

# Log infos
echo "Running template slurm job "
pwd; hostname; date

# Loading conda environment
# module load Anaconda3
# conda activate env_name

# (alternatively)
source ~/.bashrc
source activate env_name

# Checking GPU availability
echo "GPU is available: " && python -c 'import torch; print(torch.cuda.is_available())'

# Running program
srun python /src/hello_world_module.py
