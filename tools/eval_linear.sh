#!/bin/bash
#SBATCH --account retinocvd
#SBATCH --mem 128g
#SBATCH -c 16
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH --time 48:00:00

cd /home/livieymli/retidino
/home/livieymli/miniforge3/envs/cu39/bin/python -m torch.distributed.launch --use_env --nproc_per_node=4 dino/eval_linear.py --pretrained_weights results/vanilla/checkpoint.pth --data_path aptos/train_images --output_dir results/eval_linear --num_labels 5
