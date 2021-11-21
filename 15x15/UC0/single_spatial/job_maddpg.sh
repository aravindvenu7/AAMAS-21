#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64000
#SBATCH -t 0-6:00
#SBATCH --open-mode=append
#SBATCH -o jobIO.out
#SBATCH -e jobE.err
module load python
module load Anaconda3/2019.10
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01
source activate poaching
python train_maddpg.py