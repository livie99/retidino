#!/bin/bash
#SBATCH --account retinocvd
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --time 32:00:00

cd /home/livieymli/retidino
/home/livieymli/miniforge3/envs/cu39/bin/python src/customised_dataset.py