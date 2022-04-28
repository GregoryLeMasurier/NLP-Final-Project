# Final Project - Gregory LeMasurier and Mojtaba Talaei Khoei

# Dependencies: pip install rouge-score nltk sentencepiece

import os
import random
import transformers
from transformers import PegasusTokenizer, PegasusConfig
from transformers import PegasusForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import datasets
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import wandb
from packaging import version
from tqdm.auto import tqdm
from copy import deepcopy
import logging
import bitsandbytes as bnb

logger = logging.getLogger("Summarization")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def sample_small_debug_dataset(raw_datasets):
    random_indices = random.sample(list(range(len(raw_datasets["train"]))), 100)
    subset = raw_datasets["train"].select(random_indices)
    raw_datasets["train"] = deepcopy(subset)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = deepcopy(subset)
    if "test" in raw_datasets:
        raw_datasets["test"] = deepcopy(subset)
    return raw_datasets

def sample_small_eval_dataset(raw_datasets):
    val_random_indices = random.sample(list(range(len(raw_datasets["validation"]))), 3000)
    val_subset = raw_datasets["validation"].select(val_random_indices)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = deepcopy(val_subset)
   
    return raw_datasets

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()

rouge = datasets.load_metric("rouge")

cpu_only = False

dataset_name = 'cnn_dailymail'
dataset_version = '3.0.0'
wandb_project = "PegasusSummarization"
output_dir = "output_dir/"
device = 'cuda' if (torch.cuda.is_available() and not cpu_only) else 'cpu'

print("DEVICE: " + str(device) + "\n\n")

if torch.cuda.is_available:
    torch.cuda.empty_cache()

model_name = 'google/pegasus-xsum'
tokenizer_name = 'google/pegasus-xsum' #'google/pegasus-cnn_dailymail'
seq_len = 512
batch_size = 8
learning_rate = 5e-5
weight_decay = 0.0
num_train_epochs = 2
lr_scheduler_type = "linear"
num_warmup_steps = 0
eval_every_steps = 30000
k = int(seq_len * 0.3)
accum_iter = 4
#out_dim = 4096

# Flag to use smaller sample
debug = False
smallEval = False

def main():
    torch.cuda.empty_cache()

    logger.info(f"Starting tokenizer training")

    logger.info(f"Loading dataset")

    run = wandb.init(project=wandb_project) #Skipping config for now - will add back later

    os.makedirs(output_dir, exist_ok=True)

    raw_datasets = load_dataset(dataset_name, dataset_version)

    print(raw_datasets)

    # Make a small dataset for proof of concept
    if debug:
        raw_datasets = sample_small_debug_dataset(raw_datasets)
    elif smallEval:
        raw_datasets = sample_small_eval_dataset(raw_datasets)

    ## TOKENIZER
    tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)
    ## RANDOM MODEL
    config = PegasusConfig(
        use_cache=False,
        max_position_embeddings=seq_len,
        vocab_size=tokenizer.vocab_size
    )
    model = PegasusForConditionalGeneration(config).to(device)

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        inputs = [ex for ex in examples['article']]
        targets = [ex for ex in examples['highlights']]
        model_inputs = tokenizer(inputs, max_length=seq_len, truncation=True)
        model_inputs['labels'] = tokenizer(targets, max_length=seq_len, truncation=True)['input_ids']
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Tokenizing the dataset",
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"] if "validaion" in tokenized_datasets else tokenized_datasets["test"]
    test_dataset = tokenized_datasets["test"]


    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Sample {index} of the training set input ids: {train_dataset[index]['input_ids']}.")
        logger.info(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        logger.info(f"Decoded labels: {tokenizer.decode(train_dataset[index]['labels'])}")
        logger.info("\n")

    collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=seq_len, padding='max_length', label_pad_token_id=0)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_size=batch_size
    )

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = transformers.get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))

    #batch = next(iter(train_dataloader))

    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        batch_index = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = out["loss"]
            logits = out["logits"]
            res = torch.topk(logits, k=k)
            values = res[0]

            loss.backward()

            if ((batch_index + 1) % accum_iter == 0) or (batch_index + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1
            batch_index += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if (global_step % eval_every_steps == 0) or (global_step >= max_train_steps):
                model.eval()

                generations = []
                eval_labels = []
                eval_progress_bar = tqdm(range(len(eval_dataloader)))
                for batch in eval_dataloader:
                    eval_input_ids = batch["input_ids"].to(device)
                    eval_labels.append(batch["labels"].to(device))
                    encoded_summary = model.generate(eval_input_ids)
                    generations.append(encoded_summary)
                    eval_progress_bar.update(1)

                rouge_score = rouge.compute(predictions=generations, references=eval_labels)

                metric = {}
                for rouge_type in rouge_score:
                    metric['eval/' + rouge_type + "/precision"] = rouge_score[rouge_type][1][0]
                    metric['eval/' + rouge_type + "/recall"] = rouge_score[rouge_type][1][1]
                    metric['eval/' + rouge_type + "/f1-score"] = rouge_score[rouge_type][1][2]

                wandb.log(metric, step=global_step)

                logger.info("Saving model checkpoint to %s", output_dir)
                model.save_pretrained(output_dir)

                model.train()

            if global_step >= max_train_steps:
                break

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=batch_size
    )

    summaries = []
    test_labels = []
    for batch in test_dataloader:
        test_input_ids = batch["input_ids"].to(device)
        test_labels.append(batch["labels"].to(device))
        test_encoded_summary = model.generate(test_input_ids)
        summaries.append(test_encoded_summary)
        decoded_summaries = tokenizer.batch_decode(test_encoded_summary, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("Summary: " + str(decoded_summaries))
    run.finish()  # stop wandb run

if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
