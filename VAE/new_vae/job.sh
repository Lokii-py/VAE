#!/bin/bash

#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --partition=gpu                                  #name of the partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                                        #nodes requested
#SBATCH --cpus-per-task=4                               #no. of cpu's per task
#SBATCH --ntasks=1                                       #no. of tasks (cpu cores)
#SBATCH --mem=32G                                        #memory requested
#SBATCH --job-name=vae                                 #job name
#SBATCH --time=48:00:00                                  #time limit in the form days-hours:minutes
#SBATCH --output=vae_cifar_%j.out                             #out file with unique jobid
#SBATCH --error=vae_cifar_%j.err                              #err file with unique jobid
#SBATCH --mail-type=BEGIN,FAIL,END                       #email sent to user during begin,fail and end of job
#SBATCH --mail-user=$ldd3g@umsystem.edu                  #email id to be sent to(please change your email id)
# echo "### Starting at: $(date) ###"
# loading module
module load anaconda

source $HOME/.bashrc

# Debugging: Print environment details
echo "Environment activated"
which python
python --version


python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA current device:', torch.cuda.current_device()); print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))"

python main.py --dataset cifar10 --train 