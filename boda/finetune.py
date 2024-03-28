import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset, concatenate_datasets
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, AutoModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from arguments import ModelArguments, DataArguments
import wandb
logger = logging.getLogger(__name__)



# Define a function to preprocess data for the model
def tokenize_function(examples, tokenizer):
    ### TODO: Implement this function
    return tokenized

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    wandb.init(project=f"nlp702_project_{model_args.model_name_or_path}_{data_args.dataset}" )

    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    ## load model
    model = AutoModel.from_pretrained(model_args.model_name_or_path)

    dialects = [ds.strip() for ds in data_args.dataset_configs.split(",")] if data_args.dataset_configs is not None else []

    # Load the dataset
    datasets = []
    for dialect in dialects:
        datasets.append( load_dataset(data_args.dataset,dialect))

    train_dataset = concatenate_datasets([x['train'] for x in datasets]).shuffle(seed=42)
    val_dataset = concatenate_datasets([x['validation'] for x in datasets]).shuffle(seed=42)
    test_dataset = concatenate_datasets([x['test'] for x in datasets]).shuffle(seed=42)

    # Preprocess training and validation data
    train_dataset = train_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)
    val_dataset = val_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)
    test_dataset = test_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)


    save_path = f'{training_args.save_dir}/{model_args.model_name_or_path}'
    training_args.output_dir = save_path

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=val_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)
    # Start training
    trainer.train()
    


    # Save the fine-tuned model
    trainer.save_model(f"{save_path}/best")  # Adjust save directory
    eval_results = trainer.evaluate(test_dataset)

    print("Evaluation Results:", eval_results)

if __name__ == "__main__":
    main()