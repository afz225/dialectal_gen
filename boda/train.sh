#!/bin/bash

#SBATCH --job-name=NLP702-Assignment2 # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=48G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-017


SAVEDIR="/l/users/$USER/nlp702-assignment"
HF_CACHE_DIR="/l/users/$USER/hugging_face"

################################# Wikilingua ##########################
# python finetune.py \
# --output_dir=$SAVEDIR \
# --wandb_project="nlp702-project-t5-base-wikilingua-5dialects" \
# --dataset_configs aus,nig,wel,col,hon \
# --model_name_or_path='google/flan-t5-base' \
# --cache_dir ${HF_CACHE_DIR} \
# --dataset ashabrawy/dia_wikilingua \
# --do_train \
# --do_eval \
# --load_best_model_at_end True \
# --evaluation_strategy steps \
# --num_train_epochs=10 \
# --save_steps=200 \
# --eval_steps=200 \
# --logging_steps=100 \
# --report_to="all" \
# --per_device_train_batch_size=8 \
# --per_device_eval_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --predict_with_generate True
# echo "ending "

################################# Copa ##########################
# --dataset_configs aus,nig,wel,col,hon \
python finetune.py \
--output_dir=$SAVEDIR \
--wandb_project="nlp702-project-xlm-roberta-6dialects-figqa_ahmed_preprocess" \
--model_name_or_path='FacebookAI/xlm-roberta-base' \
--cache_dir ${HF_CACHE_DIR} \
--dataset ashabrawy/dia_figqa \
--do_train \
--do_eval \
--load_best_model_at_end True \
--evaluation_strategy steps \
--num_train_epochs=200 \
--save_steps=200 \
--eval_steps=100 \
--logging_steps=100 \
--report_to="all" \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--metric_for_best_model accuracy \
--warmup_ratio=0.07 \
--weight_decay=0.01 \
--learning_rate=1e-06
echo "ending "
# --predict_with_generate True
# --lr_scheduler_type="linear" 