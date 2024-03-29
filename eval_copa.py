import argparse
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (RobertaForMultipleChoice, RobertaTokenizer, Trainer,
                          TrainingArguments, XLMRobertaForMultipleChoice,
                          XLMRobertaTokenizer)

# Create the parser
parser = argparse.ArgumentParser(description='A test script for argparse.')

# Add arguments
parser.add_argument('--dataset', required=True,type=str, help='Which dataset used')
parser.add_argument('--model', required=True, type=str, help='Model used.')
parser.add_argument('--language', required=True, type=str, help='Language used.')
parser.add_argument('--logging_dir', required=True, type=str, help='Directory for saving the models.')

# Parse arguments
args = parser.parse_args()

# Use arguments
dataset = args.dataset
model_name = args.model
language = args.language
logging_dir = args.logging_dir

print(model_name, dataset)

metric = evaluate.load("accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
data = load_dataset(dataset, language)

if model_name == "roberta-base":
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
elif model_name == "xlm-roberta-base":
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
else:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    print("Using the default roberta tokenizer, be careful")

output_dir = f"{logging_dir}/{model_name}"

def preprocess_function(examples):
    # Unpack the premises and choices
    premises = examples['premise']
    choices_1 = examples['choice1']
    choices_2 = examples['choice2']
    labels = examples['label']

    # Tokenize premises and choices
    # Note that we provide both choices together as multiple_choices_inputs
    multiple_choices_inputs = []
    for premise, choice1, choice2 in zip(premises, choices_1, choices_2):
        multiple_choices_inputs.append(tokenizer.encode_plus( \
            premise, choice1, max_length=512, padding='max_length', \
            truncation=True))
        multiple_choices_inputs.append(tokenizer.encode_plus( \
            premise, choice2, max_length=512, padding='max_length', \
            truncation=True))

    # RoBERTa expects a list of all first choices and a list of all second 
    # choices, hence we restructure the inputs
    input_ids = [x['input_ids'] for x in multiple_choices_inputs]
    attention_masks = [x['attention_mask'] for x in multiple_choices_inputs]

    # Restructure inputs to match the expected format for RobertaForMultipleChoice
    features = {
        'input_ids': torch.tensor(input_ids).view(-1, 2, 512),
        'attention_mask': torch.tensor(attention_masks).view(-1, 2, 512),
        'labels': torch.tensor(labels)
    }
    return features

# Map the preprocessing function over the dataset
tokenized_datasets = data.map(preprocess_function, batched=True)

# Load the pre-trained RobertaForMultipleChoice model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == "roberta-base":
    model = RobertaForMultipleChoice.from_pretrained(f"{output_dir}/best").to(device)
elif model_name == "xlm-roberta-base":
    model = XLMRobertaForMultipleChoice.from_pretrained(f"{output_dir}/best").to(device)
else:
    model = RobertaForMultipleChoice.from_pretrained(f"{output_dir}/best").to(device)
    print("Using the default roberta, be careful")

# Optional: Evaluate the best model again for confirmation, using the Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=f"{output_dir}/best",  # Ensure this matches where you're saving the model
        per_device_eval_batch_size=8,
    ),
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate(tokenized_datasets['validation'])
print("Final Evaluation on Best Model: ", language, " ", eval_results)