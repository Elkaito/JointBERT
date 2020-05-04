#!/usr/bin/bash
#
#SBATCH --job-name=atis
#SBATCH --output=/ukp-storage-1/tanaka/JB-Kshot/seedAtis/preTrain.txt
#SBATCH --mail-user=kai-tanaka@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1


JOINTBERT=/ukp-storage-1/tanaka/JB-Kshot

source /ukp-storage-1/tanaka/JB-Kshot/my_venv/bin/activate
module purge
module load cuda/10.0

# tasks can be: atis, snips, facebook, fb-alarm, fb-reminder, fb-weather
K=(1 2 4 8 10 50)
PRETASKS=(snips fb-weather fb-reminder fb-alarm)
SEEDS=(1 26 60 123)

for PRETASK in ${PRETASKS[@]};
do
  for K in ${K[@]};
    do
     for SEED in ${SEEDS[@]};
            do

            echo $TASK $K $PRETASK $SEED
            RESULTS=$JOINTBERT/results/$TASK/$K/$PRETASK/$SEED

            rm -rf $RESULTS
            mkdir -p $RESULTS

            preModelDir = "${PRETASK}_model"
            python /ukp-storage-1/tanaka/JB-Kshot/main.py --task atis\
                  --pre_task $PRETASK \
                  --model_type bert \
                  --model_dir $RESULTS \
                  --do_train --do_eval \
                  --pre_model_dir  \
                  --K $K \
                  --seed $SEED \
     done
  done
done



