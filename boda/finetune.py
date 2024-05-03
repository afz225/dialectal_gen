import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback
import dataclasses
import logging
import os
import evaluate
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset, concatenate_datasets,Value
import numpy as np
from typing import Union, Optional
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
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
from nltk.tokenize import sent_tokenize
import nltk
from evaluate import load

from transformers import DataCollatorForSeq2Seq,RobertaForMultipleChoice,RobertaTokenizer
nltk.download("punkt")
logger = logging.getLogger(__name__)
from transformers import (RobertaForMultipleChoice, RobertaTokenizer, Trainer,
                          TrainingArguments, XLMRobertaForMultipleChoice,
                          XLMRobertaTokenizer)


# Define a function to preprocess data for the model
def tokenize_dia_wikilingua(examples, tokenizer):
    # inputs = ['summarize: '+ example for example in examples["source"]]
    # model_inputs = tokenizer(inputs, padding='max_length', truncation=True)
    # labels = tokenizer(text_target=examples["target"], padding="max_length", truncation=True)
    # labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs

    max_input_length = 1024
    max_target_length = 128
    prefix = "summarize: "    
    inputs = [prefix + doc for doc in examples["source"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs    


def tokenize_dia_copa(examples,tokenizer):
    # Unpack the premises and choices
    premise_index = "premise" if "premise" in examples else "startphrase"
    choice_1_index = "choice1" if "choice1" in examples else "ending1"
    choice_2_index = "choice2" if "choice2" in examples else "ending2"
    label_index = "label" if "label" in examples else "labels"

    premises = examples[premise_index]
    choices_1 = examples[choice_1_index]
    choices_2 = examples[choice_2_index]
    labels = examples[label_index]

    # # Tokenize premises and choices
    # # Note that we provide both choices together as multiple_choices_inputs
    # multiple_choices_inputs = []
    # for premise, choice1, choice2 in zip(premises, choices_1, choices_2):
    #     multiple_choices_inputs.append(tokenizer.encode_plus( \
    #         premise, choice1, max_length=512, padding='max_length', \
    #         truncation=True))
    #     multiple_choices_inputs.append(tokenizer.encode_plus( \
    #         premise, choice2, max_length=512, padding='max_length', \
    #         truncation=True))

    # # RoBERTa expects a list of all first choices and a list of all second 
    # # choices, hence we restructure the inputs
    # input_ids = [x['input_ids'] for x in multiple_choices_inputs]
    # attention_masks = [x['attention_mask'] for x in multiple_choices_inputs]

    # # Restructure inputs to match the expected format for RobertaForMultipleChoice
    # features = {
    #     'input_ids': torch.tensor(input_ids).view(-1, 2, 512),
    #     'attention_mask': torch.tensor(attention_masks).view(-1, 2, 512),
    #     'labels': torch.tensor(labels)
    # }
    # return features


    ############################### Ahmed's code ########################################
    # Repeat each prompt for 5 times to go with the 5 possibilities of each option
    first_sentences = [[context] * 2 for context in examples[premise_index]]
    # Grab all options
    second_sentences = [[ending1, examples[choice_2_index][i]] for i, ending1 in enumerate(examples[choice_1_index])]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {
        k: [v[i : i + 2] for i in range(0, len(v), 2)]
        for k, v in tokenized_examples.items()
    }


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    wandb.init(project=model_args.wandb_project)

    ## load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # model = AutoModel.from_pretrained(model_args.model_name_or_path)


    datasets = []
    if data_args.dataset.split('/')[1] == "dia_wikilingua":
        print("Training for WikiLingua")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        metric = load("rouge")
        compute_metrics_fn = f'compute_metrics_dia_wikilingua'
        map_fn = f'tokenize_dia_wikilingua'
        init_dataset = load_dataset('GEM/wiki_lingua')

        ## Here the dataset reference has type list, so we need to select the first element, to be able to concat it with dialects dataset 
        init_dataset['train'] = init_dataset['train'].select(range(500)).map(lambda example: {"references": example["references"][0]})
        init_dataset['validation'] = init_dataset['validation'].select(range(100)).map(lambda example: {"references": example["references"][0]})
        init_dataset['test'] = init_dataset['test'].select(range(100)).map(lambda example: {"references": example["references"][0]})

        datasets.append(init_dataset)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,label_pad_token_id=-100,pad_to_multiple_of=8)

        # trainer = Seq2SeqTrainer

    elif data_args.dataset.split('/')[1] in  ["dia_copa",'dia_figqa']:

        if model_args.model_name_or_path == "roberta-base":
            model = RobertaForMultipleChoice.from_pretrained(model_args.model_name_or_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)

        elif model_args.model_name_or_path == "xlm-roberta-base" or model_args.model_name_or_path == 'FacebookAI/xlm-roberta-base':
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_args.model_name_or_path)
            model = XLMRobertaForMultipleChoice.from_pretrained(model_args.model_name_or_path)


        if data_args.dataset.split('/')[1] == "dia_copa":
            print("Training for COPA")
    
            init_dataset = load_dataset("super_glue", "copa")
            ### Cast features again 
            new_features = init_dataset['train'].features.copy()
            new_features["label"] = Value("int64")
            new_features["idx"] = Value("int64")
            init_dataset['train'] = init_dataset['train'].cast(new_features)
            init_dataset['validation'] = init_dataset['train'].cast(new_features)
            # init_dataset['test'] = init_dataset['train'].cast(new_features)

            datasets.append(init_dataset)

        elif data_args.dataset.split('/')[1] == "dia_figqa":
            print("Training for FIGQA")
            init_dataset = load_dataset("nightingal3/fig-qa")
            datasets.append(init_dataset)

        metric = load("accuracy")
        compute_metrics_fn = f'compute_metrics_dia_copa'
        map_fn = f'tokenize_dia_copa'




    ## load model

    dialects = [ds.strip() for ds in data_args.dataset_configs.split(",")] if data_args.dataset_configs is not None else []

    # Load the dataset
    for dialect in dialects:
        ## Load dialects datasets and remove the Unnamed: 0 column
        datasets.append( load_dataset(data_args.dataset,dialect).remove_columns(['Unnamed: 0']))

    train_dataset = concatenate_datasets([x['train'].select(range(500)) for x in datasets])#.shuffle()
    val_dataset = concatenate_datasets([x['validation'].select(range(500)) for x in datasets])#.shuffle()
    # test_dataset = concatenate_datasets([x['test'] for x in datasets]).shuffle(seed=42)

    # ds_name = data_args.dataset.split("/")[1]
    # map_fn = f'tokenize_{ds_name}'

    # Preprocess training and validation data
    train_dataset = train_dataset.map(eval(map_fn),fn_kwargs={"tokenizer":tokenizer}, batched=True)
    val_dataset = val_dataset.map(eval(map_fn),fn_kwargs={"tokenizer":tokenizer}, batched=True)
    # test_dataset = test_dataset.map(eval(map_fn),fn_kwargs={"tokenizer":tokenizer}, batched=True)


    save_path = f'{training_args.output_dir}/{model_args.model_name_or_path}'
    training_args.output_dir = save_path

    def compute_metrics_dia_wikilingua(eval_preds):
        print('hi')
        print("eval_preds",eval_preds)
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
        # return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'gen_len': 0.0}

    def compute_metrics_dia_copa(eval_pred):

        """
        Compute rouge and bleu metrics for seq2seq model generated prediction.

        tip: we can run trainer.predict on our eval/test dataset to see what a sample
        eval_pred object would look like when implementing custom compute metrics function
        """
        predictions, labels = eval_pred
        # Decode generated summaries, which is in ids into text
        _, predictions = torch.max(torch.tensor(predictions), dim=1)
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        return clf_metrics.compute(predictions=predictions, references=labels)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=val_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=eval(compute_metrics_fn),
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer) if data_args.dataset.split('/')[1] in  ["dia_copa",'dia_figqa'] else DataCollatorForSeq2Seq(tokenizer, model=model,label_pad_token_id=-100,pad_to_multiple_of=8),
    # compute_metrics=compute_metrics_dia_wikilingua,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)
    # Start training
    trainer.train()
    


    # Save the fine-tuned model
    trainer.save_model(f"{save_path}/best")  # Adjust save directory
    eval_results = trainer.evaluate(val_dataset)


    print("Evaluation Results:", eval_results)
    wandb.log(eval_results)

if __name__ == "__main__":
    main()