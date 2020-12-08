#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --exclusive
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=yh1n19@soton.ac.uk
export STUDENT_ID=$(whoami)

export DATA_HOME=/scratch/${STUDENT_ID}

export DATA_PATH=${DATA_HOME}/data/material

conda init

source /local/software/conda/miniconda-py3-new/bin/activate cv-pytorch

conda activate cv-pytorch

cd ..

python main.py --data-root "${DATA_PATH}" --stage patch --batch-size 128