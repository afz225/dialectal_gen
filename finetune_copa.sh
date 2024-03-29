#!/bin/bash

#SBATCH --job-name=eval_copa
#SBATCH --error=/home/daria.kotova/nlp702/project/dialectal_gen/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/nlp702/project/dialectal_gen/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l4-006


echo "starting......................."
###################### RUN Eval ######################

python finetune_copa.py \
--dataset="copa" \
--model="xlm-roberta-base" \
--logging_dir="new_hyperparams"


echo " ending "