#!/bin/bash
#
#SBATCH --job-name=atis10%
#SBATCH --output=/ukp-storage-1/tanaka/JB-Kshot/atisResult/atis10%.txt
#SBATCH --mail-user=kai-tanaka@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/tanaka/JB-Kshot/my_venv/bin/activate
module purge
module load cuda/10.0
python /ukp-storage-1/tanaka/JB-Kshot/main.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval \

