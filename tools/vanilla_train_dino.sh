#!/bin/bash
#SBATCH --account retinocvd
#SBATCH --mem 128g
#SBATCH -c 16
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --time 72:00:00

cd /home/livieymli/retidino
/home/livieymli/miniforge3/envs/cu39/bin/python -m torch.distributed.launch --use_env --nproc_per_node=1 dino/main_dino.py --arch vit_small --data_path aptos/train_images/train --output_dir results/vanilla