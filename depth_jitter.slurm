#!/bin/bash

#SBATCH --time=07:00:00
#SBATCH --job-name=res-fn-DJ+CJ
#SBATCH --partition=besteffort
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /mundus/mrahman528/q2l_infer6.log
#SBATCH -e /mundus/mrahman528/q2l_err_infer6.log
#SBATCH --gres=gpu:a40-48:2

#cd Depth-Jitter
#source ~/miniconda3/etc/profile.d/conda.sh
conda activate query2l
python train_q2l.py --dataset FathomNet
