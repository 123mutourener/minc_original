#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --nodes=2
#SBATCH --time=60:00:00
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=yh1n19@soton.ac.uk
# auto-saving
#SBATCH --signal=SIGUSR1@90

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export STUDENT_ID=$(whoami)

export DATA_HOME=/scratch/${STUDENT_ID}

export DATA_PATH=${DATA_HOME}/data/material

conda init

source /local/software/conda/miniconda-py3-new/bin/activate cv-pytorch

conda activate cv-pytorch

cd ..

srun python light_main.py --data-root "${DATA_PATH}" --stage patch --batch-size 256 --tag random_sample_256 --gpus 4 --num-nodes 2 --epochs 10