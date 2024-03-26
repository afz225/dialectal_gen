#!/bin/bash

#SBATCH -n 100
#SBATCH -t 48:00:00

export HF_ACCESS_TOKEN=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
export TRANSFORMERS_CACHE=/scratch/afz225/.cache

python3 gen_record.py

