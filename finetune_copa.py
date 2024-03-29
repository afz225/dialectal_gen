print("Started importing")
from datasets import load_dataset
import argparse
import torch
import evaluate
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForMultipleChoice, \
    XLMRobertaTokenizer, XLMRobertaForMultipleChoice, Trainer, TrainingArguments

# Create the parser
parser = argparse.ArgumentParser(description='A test script for argparse.')

# Add arguments
parser.add_argument('--dataset', required=True,type=str, help='Which dataset used.')
parser.add_argument('--model', required=True, type=str, help='Model used.')
parser.add_argument('--logging_dir', required=True, type=str, help='Directory for saving the models.')

# Parse arguments
args = parser.parse_args()

# Use arguments
dataset = load_dataset("super_glue", args.dataset)
model_name = args.model
logging_dir = args.logging_dir

print(model_name, dataset)

metric = evaluate.load("accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if model_name == "roberta-base":
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
elif model_name == "xlm-roberta-base":
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
else:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    print("Using the default roberta tokenizer, be careful")


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
        multiple_choices_inputs.append(tokenizer.encode_plus(premise, choice1, max_length=512, padding='max_length', truncation=True))
        multiple_choices_inputs.append(tokenizer.encode_plus(premise, choice2, max_length=512, padding='max_length', truncation=True))

    # RoBERTa expects a list of all first choices and a list of all second choices, hence we restructure the inputs
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
tokenized_datasets = dataset.map(preprocess_function, batched=True)

if model_name == "roberta-base":
    model = RobertaForMultipleChoice.from_pretrained(model_name).to(device)
elif model_name == "xlm-roberta-base":
    model = XLMRobertaForMultipleChoice.from_pretrained(model_name).to(device)
else:
    model = RobertaForMultipleChoice.from_pretrained(model_name).to(device)
    print("Using the default roberta, be careful")

output_dir = f"{logging_dir}/{model_name}"

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=50,
    per_device_train_batch_size=8,
    warmup_steps=10, 
    weight_decay=0.01, 
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=1e-6,
    save_steps=50, 
    save_total_limit=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Path where the checkpoints are saved
checkpoints_path = output_dir
checkpoints = [os.path.join(checkpoints_path, name) for name in os.listdir(checkpoints_path) if name.startswith("checkpoint")]

# Placeholder for the best performance
best_performance = 0.0
best_checkpoint = None

for checkpoint in checkpoints:
    # Load the model from checkpoint
    if model_name == "roberta-base":
        model = RobertaForMultipleChoice.from_pretrained(checkpoint).to(device)
    elif model_name == "xlm-roberta-base":
        model = XLMRobertaForMultipleChoice.from_pretrained(checkpoint).to(device)
    else:
        model = RobertaForMultipleChoice.from_pretrained(checkpoint).to(device)
        print("Using the default roberta, be careful")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=8,  # Adjust as necessary
        ),
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    eval_results = trainer.evaluate(tokenized_datasets['validation'])

    # Assuming 'accuracy' is your metric of interest
    print(eval_results)
    performance = eval_results["eval_accuracy"]

    # Update the best checkpoint if current model is better
    if performance > best_performance:
        best_performance = performance
        best_checkpoint = checkpoint

print(f"Best checkpoint: {best_checkpoint} with Eval Loss: {best_performance}")
# model = RobertaForMultipleChoice.from_pretrained(f"/home/george.ibrahim/Downloads/Semester 2/NLP702/Project/{model_name}_results_{language}/best").to(device)

if best_checkpoint:
    print(f"Best checkpoint: {best_checkpoint} with Eval Loss: {best_performance}")

    # Load the best model
    if model_name == "roberta-base":
        best_model = RobertaForMultipleChoice.from_pretrained(best_checkpoint).to(device)
    elif model_name == "xlm-roberta-base":
        best_model = XLMRobertaForMultipleChoice.from_pretrained(best_checkpoint).to(device)
    else:
        best_model = RobertaForMultipleChoice.from_pretrained(best_checkpoint).to(device)
        print("Using the default roberta, be careful")

    # Directly save the best model to the desired directory
    best_model.save_pretrained(f"{output_dir}/{model_name}/best")

    # If you want to save the tokenizer as well
    tokenizer.save_pretrained(f"{output_dir}/{model_name}/best")

    # Optional: Evaluate the best model again for confirmation, using the Trainer
    trainer = Trainer(
        model=best_model,
        args=TrainingArguments(
            output_dir=f'./{output_dir}/{model_name}/best',  # Ensure this matches where you're saving the model
            per_device_eval_batch_size=8,
        ),
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(tokenized_datasets['validation'])
    print("Final Evaluation on Best Model:", eval_results)
else:
    print("No best checkpoint identified.")
