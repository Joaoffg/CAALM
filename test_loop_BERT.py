<<<<<<< HEAD
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import gc

# Define datasets, sample sizes, random seeds, and models
datasets_list = ["Hate", "Corona", "Military", "Morality"]
sample_sizes_list = [100, 500, 1000, 2500, 5000, 10000, 25000, 'full']
seeds_list = [42, 43, 44, 45, 46]
models_list = ["microsoft/deberta-v3-base", "facebook/roberta-large"]

# Mapping from dataset names to file paths
dataset_paths = {
    "Morality": {
        "train": "Morality/df_manifesto_morality_train_w_context.csv",
        "test": "Morality/df_manifesto_morality_test_w_context.csv",
    },
    "Corona": {
        "train": "Corona/df_coronanet_20220124_train_w_context.csv",
        "test": "Corona/df_coronanet_20220124_test_w_context.csv",
    },
    "Military": {
        "train": "Military/df_manifesto_military_train_w_context.csv",
        "test": "Military/df_manifesto_military_test_w_context.csv",
    },
    "Hate": {
        "train": "Hate/df_hate_train_w_context.csv",
        "test": "Hate/df_hate_test_w_context.csv",
    },
}

# Function to preprocess data based on the dataset
def preprocess_data(df_train, df_test, dataset_name):
    if dataset_name in ["Military", "Morality"]:
        # Run for Military and Morality datasets, with context
        df_train["text_prepared"] = df_train.text_preceding.fillna("") + '. The quote: "' + df_train.text_original.fillna("") + '" - end of the quote. ' + df_train.text_following.fillna("") + df_train.context.fillna("")
        df_test["text_prepared"] = df_test.text_preceding.fillna("") + '. The quote: "' + df_test.text_original.fillna("") + '" - end of the quote. ' + df_test.text_following.fillna("") + df_test.context.fillna("")
    elif dataset_name in ["Corona", "Hate"]:
        # Run for Corona and Hate datasets, with context
        df_train["text_prepared"] = df_train.text.fillna("") + " " + df_train.context.fillna("")
        df_test["text_prepared"] = df_test.text.fillna("") + " " + df_test.context.fillna("")
    # Run for all datasets, with context
    df_train['text_prepared'] = df_train['text_prepared'].str.replace('<pad>', '', regex=False)
    df_test['text_prepared'] = df_test['text_prepared'].str.replace('<pad>', '', regex=False)
    return df_train, df_test

# Function to clean GPU memory
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Define the compute_metrics function
def compute_metrics(eval_pred, num_labels):
    predictions, labels = eval_pred
    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    preds = preds.astype(int)
    labels = labels.astype(int)
    y_true = labels
    y_pred = preds

    # Compute metrics
    eval_f1_macro = f1_score(y_true, y_pred, average='macro')
    eval_f1_micro = f1_score(y_true, y_pred, average='micro')
    eval_accuracy_balanced = balanced_accuracy_score(y_true, y_pred)
    eval_accuracy_not_b = accuracy_score(y_true, y_pred)

    return {
        'eval_f1_macro': eval_f1_macro,
        'eval_f1_micro': eval_f1_micro,
        'eval_accuracy_balanced': eval_accuracy_balanced,
        'eval_accuracy_not_b': eval_accuracy_not_b
    }

# File to save results
results_file = "results.txt"

# Open the results file in append mode
with open(results_file, "a") as f_results:
    # Loop over models
    for model_name in models_list:
        # Loop over datasets
        for dataset_name in datasets_list:
            # Determine the number of labels based on the dataset
            if dataset_name == "Corona":
                num_labels = 20
            else:
                num_labels = 3

            # Loop over sample sizes
            for sample_size in sample_sizes_list:
                # Loop over random seeds
                for SEED_GLOBAL in seeds_list:
                    print(f"Starting iteration with model: {model_name}, dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}")
                    # Clean GPU memory before each iteration
                    clean_memory()

                    # Set random seed for reproducibility
                    np.random.seed(SEED_GLOBAL)
                    torch.manual_seed(SEED_GLOBAL)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED_GLOBAL)

                    # Load data
                    train_path = dataset_paths[dataset_name]["train"]
                    test_path = dataset_paths[dataset_name]["test"]
                    df_train = pd.read_csv(train_path, index_col="idx")
                    df_test = pd.read_csv(test_path, index_col="idx")

                    print(f"Loaded data for {dataset_name}. Train size: {len(df_train)}, Test size: {len(df_test)}.")

                    # Sample training data
                    if sample_size != 'full':
                        df_train = df_train.sample(n=min(int(sample_size), len(df_train)), random_state=SEED_GLOBAL).copy(deep=True)
                        print(f"Sampled training data to size: {len(df_train)}.")
                    else:
                        print("Using full training data.")

                    # Preprocess data
                    df_train, df_test = preprocess_data(df_train, df_test, dataset_name)

                    # Convert pandas dataframes to Hugging Face dataset object
                    dataset = datasets.DatasetDict({
                        "train": datasets.Dataset.from_pandas(df_train),
                        "test": datasets.Dataset.from_pandas(df_test)
                    })

                    # Tokenize
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

                    # Use GPU (cuda) if available, otherwise use CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Device: {device}")
                    model.to(device);

                    # Tokenize function
                    def tokenize(examples):
                        return tokenizer(examples["text_prepared"], truncation=True, max_length=512)

                    dataset["train"] = dataset["train"].map(tokenize, batched=True)
                    dataset["test"] = dataset["test"].map(tokenize, batched=True)

                    # Training arguments
                    training_directory = f"{model_name.replace('/', '-')}-{dataset_name}-{sample_size}-{SEED_GLOBAL}"
                    train_args = TrainingArguments(
                        output_dir=f'./results/{training_directory}',
                        logging_dir=f'./logs/{training_directory}',
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=80,
                        num_train_epochs=6,
                        warmup_ratio=0.25,
                        weight_decay=0.1,
                        seed=SEED_GLOBAL,
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_accuracy_not_b",
                        fp16=torch.cuda.is_available(),
                        fp16_full_eval=torch.cuda.is_available(),
                        evaluation_strategy="no",
                        save_strategy="no",
                        report_to="none",
                    )

                    # Trainer
                    trainer = Trainer(
                        model=model,
                        tokenizer=tokenizer,
                        args=train_args,
                        train_dataset=dataset["train"],
                        eval_dataset=dataset["test"],
                        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, num_labels)
                    )

                    # Train the model
                    trainer.train()

                    # Evaluate the model
                    results = trainer.evaluate()

                    # Print and save the results
                    print(f"Results for model: {model_name}, dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}")
                    print(results)
                    # Save the results to the file
                    f_results.write(f"Model: {model_name}, Dataset: {dataset_name}, Sample size: {sample_size}, Seed: {SEED_GLOBAL}\n")
                    f_results.write(f"Results: {results}\n")
                    f_results.write("\n")
                    f_results.flush()  # Ensure data is written to file

                    # Clean up
                    del model, tokenizer, trainer, train_args, dataset, df_train, df_test
                    clean_memory()

                    print("Iteration completed.\n")
=======
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import gc

# Define datasets, sample sizes, random seeds, and models
datasets_list = ["Hate", "Corona", "Military", "Morality"]
sample_sizes_list = [100, 500, 1000, 2500, 5000, 10000, 25000, 'full']
seeds_list = [42, 43, 44, 45, 46]
models_list = ["microsoft/deberta-v3-base", "facebook/roberta-large"]

# Mapping from dataset names to file paths
dataset_paths = {
    "Morality": {
        "train": "Morality/df_manifesto_morality_train_w_context.csv",
        "test": "Morality/df_manifesto_morality_test_w_context.csv",
    },
    "Corona": {
        "train": "Corona/df_coronanet_20220124_train_w_context.csv",
        "test": "Corona/df_coronanet_20220124_test_w_context.csv",
    },
    "Military": {
        "train": "Military/df_manifesto_military_train_w_context.csv",
        "test": "Military/df_manifesto_military_test_w_context.csv",
    },
    "Hate": {
        "train": "Hate/df_hate_train_w_context.csv",
        "test": "Hate/df_hate_test_w_context.csv",
    },
}

# Function to preprocess data based on the dataset
def preprocess_data(df_train, df_test, dataset_name):
    if dataset_name in ["Military", "Morality"]:
        # Run for Military and Morality datasets, with context
        df_train["text_prepared"] = df_train.text_preceding.fillna("") + '. The quote: "' + df_train.text_original.fillna("") + '" - end of the quote. ' + df_train.text_following.fillna("") + df_train.context.fillna("")
        df_test["text_prepared"] = df_test.text_preceding.fillna("") + '. The quote: "' + df_test.text_original.fillna("") + '" - end of the quote. ' + df_test.text_following.fillna("") + df_test.context.fillna("")
    elif dataset_name in ["Corona", "Hate"]:
        # Run for Corona and Hate datasets, with context
        df_train["text_prepared"] = df_train.text.fillna("") + " " + df_train.context.fillna("")
        df_test["text_prepared"] = df_test.text.fillna("") + " " + df_test.context.fillna("")
    # Run for all datasets, with context
    df_train['text_prepared'] = df_train['text_prepared'].str.replace('<pad>', '', regex=False)
    df_test['text_prepared'] = df_test['text_prepared'].str.replace('<pad>', '', regex=False)
    return df_train, df_test

# Function to clean GPU memory
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Define the compute_metrics function
def compute_metrics(eval_pred, num_labels):
    predictions, labels = eval_pred
    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    preds = preds.astype(int)
    labels = labels.astype(int)
    y_true = labels
    y_pred = preds

    # Compute metrics
    eval_f1_macro = f1_score(y_true, y_pred, average='macro')
    eval_f1_micro = f1_score(y_true, y_pred, average='micro')
    eval_accuracy_balanced = balanced_accuracy_score(y_true, y_pred)
    eval_accuracy_not_b = accuracy_score(y_true, y_pred)

    return {
        'eval_f1_macro': eval_f1_macro,
        'eval_f1_micro': eval_f1_micro,
        'eval_accuracy_balanced': eval_accuracy_balanced,
        'eval_accuracy_not_b': eval_accuracy_not_b
    }

# File to save results
results_file = "results.txt"

# Open the results file in append mode
with open(results_file, "a") as f_results:
    # Loop over models
    for model_name in models_list:
        # Loop over datasets
        for dataset_name in datasets_list:
            # Determine the number of labels based on the dataset
            if dataset_name == "Corona":
                num_labels = 20
            else:
                num_labels = 3

            # Loop over sample sizes
            for sample_size in sample_sizes_list:
                # Loop over random seeds
                for SEED_GLOBAL in seeds_list:
                    print(f"Starting iteration with model: {model_name}, dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}")
                    # Clean GPU memory before each iteration
                    clean_memory()

                    # Set random seed for reproducibility
                    np.random.seed(SEED_GLOBAL)
                    torch.manual_seed(SEED_GLOBAL)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED_GLOBAL)

                    # Load data
                    train_path = dataset_paths[dataset_name]["train"]
                    test_path = dataset_paths[dataset_name]["test"]
                    df_train = pd.read_csv(train_path, index_col="idx")
                    df_test = pd.read_csv(test_path, index_col="idx")

                    print(f"Loaded data for {dataset_name}. Train size: {len(df_train)}, Test size: {len(df_test)}.")

                    # Sample training data
                    if sample_size != 'full':
                        df_train = df_train.sample(n=min(int(sample_size), len(df_train)), random_state=SEED_GLOBAL).copy(deep=True)
                        print(f"Sampled training data to size: {len(df_train)}.")
                    else:
                        print("Using full training data.")

                    # Preprocess data
                    df_train, df_test = preprocess_data(df_train, df_test, dataset_name)

                    # Convert pandas dataframes to Hugging Face dataset object
                    dataset = datasets.DatasetDict({
                        "train": datasets.Dataset.from_pandas(df_train),
                        "test": datasets.Dataset.from_pandas(df_test)
                    })

                    # Tokenize
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

                    # Use GPU (cuda) if available, otherwise use CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Device: {device}")
                    model.to(device);

                    # Tokenize function
                    def tokenize(examples):
                        return tokenizer(examples["text_prepared"], truncation=True, max_length=512)

                    dataset["train"] = dataset["train"].map(tokenize, batched=True)
                    dataset["test"] = dataset["test"].map(tokenize, batched=True)

                    # Training arguments
                    training_directory = f"{model_name.replace('/', '-')}-{dataset_name}-{sample_size}-{SEED_GLOBAL}"
                    train_args = TrainingArguments(
                        output_dir=f'./results/{training_directory}',
                        logging_dir=f'./logs/{training_directory}',
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=80,
                        num_train_epochs=6,
                        warmup_ratio=0.25,
                        weight_decay=0.1,
                        seed=SEED_GLOBAL,
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_accuracy_not_b",
                        fp16=torch.cuda.is_available(),
                        fp16_full_eval=torch.cuda.is_available(),
                        evaluation_strategy="no",
                        save_strategy="no",
                        report_to="none",
                    )

                    # Trainer
                    trainer = Trainer(
                        model=model,
                        tokenizer=tokenizer,
                        args=train_args,
                        train_dataset=dataset["train"],
                        eval_dataset=dataset["test"],
                        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, num_labels)
                    )

                    # Train the model
                    trainer.train()

                    # Evaluate the model
                    results = trainer.evaluate()

                    # Print and save the results
                    print(f"Results for model: {model_name}, dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}")
                    print(results)
                    # Save the results to the file
                    f_results.write(f"Model: {model_name}, Dataset: {dataset_name}, Sample size: {sample_size}, Seed: {SEED_GLOBAL}\n")
                    f_results.write(f"Results: {results}\n")
                    f_results.write("\n")
                    f_results.flush()  # Ensure data is written to file

                    # Clean up
                    del model, tokenizer, trainer, train_args, dataset, df_train, df_test
                    clean_memory()

                    print("Iteration completed.\n")
>>>>>>> 53c9b4e0ed2869a8c28af646c60a06b6f826806d
