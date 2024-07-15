#!/bin/bash
#SBATCH --account retinocvd
#SBATCH --mem 16g
#SBATCH -c 8
#SBATCH --time 8:00:00

cd /home/livieymli/retidino
/home/livieymli/miniforge3/envs/cu39/bin/python src/contour.py