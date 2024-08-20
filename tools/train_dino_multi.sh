#!/bin/bash
#SBATCH --account retinocvd
#SBATCH --mem 512g
#SBATCH -c 16
#SBATCH --partition gpu
#SBATCH --gres=gpu:8
#SBATCH --time 168:00:00

cd /home/livieymli/retidino 
/home/livieymli/miniforge3/envs/cu39/bin/python -m torch.distributed.launch --use_env --nproc_per_node=8 dino/main_dino_multi.py --arch vit_small --data_path aptos/train_images/train --output_dir results/mosaic
# When using Customized_dataset in dino_multi, the data_path can be ignored