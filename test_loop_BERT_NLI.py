<<<<<<< HEAD
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, classification_report
import gc

# Define datasets, sample sizes, random seeds, and context usage
datasets_list = ["Hate", "Corona", "Military", "Morality"]
sample_sizes_list = [100, 500, 1000, 2500, 5000, 10000, 25000, 'full']
seeds_list = [42, 43, 44, 45, 46]
use_context_options = [True, False]  # Run with and without context

# Mapping from dataset names to file paths and index columns
dataset_paths = {
    "Morality": {
        "train": "Morality/df_manifesto_morality_train_w_context.csv",
        "test": "Morality/df_manifesto_morality_test_w_context.csv",
        "index_col": "idx"
    },
    "Corona": {
        "train": "Corona/df_coronanet_20220124_train_w_context.csv",
        "test": "Corona/df_coronanet_20220124_test_w_context.csv",
        "index_col": "idx"
    },
    "Military": {
        "train": "Military/df_manifesto_military_train_w_context.csv",
        "test": "Military/df_manifesto_military_test_w_context.csv",
        "index_col": "idx"
    },
    "Hate": {
        "train": "Hate/df_hate_train_w_context.csv",
        "test": "Hate/df_hate_test_w_context.csv",
        "index_col": None  # No 'idx' column in Hate dataset
    },
}

# Hypotheses for each dataset
hypotheses_per_dataset = {
    "Hate": {
        "Offensive Language": "The quote contains offensive language, such as curse words that are not targeted at a specific group",
        "Hate Speech": "The quote contains hate speech, language that is used to express hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group",
        "Neither": "The quote does not contain hate speech or offensive language, such as hate towards a specific group or curse words"
    },
    "Corona": {
        'Anti-Disinformation Measures': "The quote is about measures against disinformation: Efforts by the government to limit the spread of false, inaccurate or harmful information",
        'COVID-19 Vaccines': "The quote is about COVID-19 vaccines. A policy regarding the research and development, or regulation, or production, or purchase, or distribution of a vaccine.",
        'Closure and Regulation of Schools': "The quote is about regulating schools and educational establishments. For example closing an educational institution, or allowing educational institutions to open with or without certain conditions.",
        'Curfew': "The quote is about a curfew: Domestic freedom of movement is limited during certain times of the day",
        'Declaration of Emergency': "The quote is about declaration of a state of national emergency",
        'External Border Restrictions': "The quote is about external border restrictions: The ability to enter or exit country borders is reduced.",
        'Health Monitoring': "The quote is about health monitoring of individuals who are likely to be infected.",
        'Health Resources': "The quote is about health resources: For example medical equipment, number of hospitals, health infrastructure, personnel (e.g. doctors, nurses), mask purchases",
        'Health Testing': "The quote is about health testing of large populations regardless of their likelihood of being infected.",
        'Hygiene': "The quote is about hygiene: Promotion of hygiene in public spaces, for example disinfection in subways or burials.",
        'Internal Border Restrictions': "The quote is about internal border restrictions: The ability to move freely within the borders of a country is reduced.",
        'Lockdown': "The quote is about a lockdown: People are obliged to shelter in place and are only allowed to leave their shelter for specific reasons",
        'New Task Force, Bureau or Administrative Configuration': "The quote is about a new administrative body, for example a new task force, bureau or administrative configuration.",
        'Public Awareness Measures': "The quote is about public awareness measures or efforts to disseminate or gather reliable information, for example information on health prevention.",
        'Quarantine': "The quote is about quarantine. People are obliged to isolate themselves if they are infected.",
        'Restriction and Regulation of Businesses': "The quote is about restricting or regulating businesses, private commercial activities: For example closing down commercial establishments, or allowing commercial establishments to open with or without certain conditions.",
        'Restriction and Regulation of Government Services': "The quote is about restricting or regulating government services or public facilities: For example closing down government services, or allowing government services to operate with or without certain conditions.",
        'Restrictions of Mass Gatherings': "The quote is about restrictions of mass gatherings: The number of people allowed to congregate in a place is limited",
        'Social Distancing': "The quote is about social distancing, reducing contact between individuals in public spaces, mask wearing.",
        "Other Policy Not Listed Above": "The quote is about something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccines, disinformation, schools, borders or travel, testing, health resources. It is not about any of these topics."
    },
    "Morality": {
        "Traditional Morality: Positive": "The quote is positive towards traditional morality, for example in favour of traditional family values, religious institutions, or against unseemly behaviour",
        "Traditional Morality: Negative": "The quote is negative towards traditional morality, for example in favour of divorce or abortion, modern families, separation of church and state, modern values",
        "Other": "The quote is not about traditional morality, for example not about family values, abortion or religion"
    },
    "Military": {
        "Military: Positive": "The quote is positive towards the military, for example for military spending, defense, military treaty obligations.",
        "Military: Negative": "The quote is negative towards the military, for example against military spending, for disarmament, against conscription.",
        "Other": "The quote is not about military or defense"
    }
}

# Function to preprocess data based on the dataset and context usage
def preprocess_data(df_train, df_test, dataset_name, use_context):
    if dataset_name in ["Military", "Morality"]:
        if use_context:
            # With context
            df_train["text_prepared"] = (
                df_train.text_preceding.fillna("") +
                '. The quote: "' + df_train.text_original.fillna("") +
                '" - end of the quote. ' + df_train.text_following.fillna("") +
                df_train.context.fillna("")
            )
            df_test["text_prepared"] = (
                df_test.text_preceding.fillna("") +
                '. The quote: "' + df_test.text_original.fillna("") +
                '" - end of the quote. ' + df_test.text_following.fillna("") +
                df_test.context.fillna("")
            )
        else:
            # Without context
            df_train["text_prepared"] = (
                df_train.text_preceding.fillna("") +
                '. The quote: "' + df_train.text_original.fillna("") +
                '" - end of the quote. ' + df_train.text_following.fillna("")
            )
            df_test["text_prepared"] = (
                df_test.text_preceding.fillna("") +
                '. The quote: "' + df_test.text_original.fillna("") +
                '" - end of the quote. ' + df_test.text_following.fillna("")
            )
    elif dataset_name in ["Corona", "Hate"]:
        if use_context:
            # With context
            df_train["text_prepared"] = df_train.text.fillna("") + " " + df_train.context.fillna("")
            df_test["text_prepared"] = df_test.text.fillna("") + " " + df_test.context.fillna("")
        else:
            # Without context
            df_train["text_prepared"] = df_train.text.fillna("")
            df_test["text_prepared"] = df_test.text.fillna("")
    # Remove '<pad>' tokens
    df_train['text_prepared'] = df_train['text_prepared'].str.replace('<pad>', '', regex=False)
    df_test['text_prepared'] = df_test['text_prepared'].str.replace('<pad>', '', regex=False)
    return df_train, df_test

# Function to clean GPU memory
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Function to format the training set for NLI
def format_nli_trainset(df_train, hypo_label_dic, random_seed=42):
    print(f"Length of df_train before formatting step: {len(df_train)}.")
    length_original_data_train = len(df_train)

    df_train_lst = []
    for label_text, hypothesis in hypo_label_dic.items():
        ## entailment
        df_train_step = df_train[df_train.label_text == label_text].copy(deep=True)
        df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
        df_train_step["label"] = [0] * len(df_train_step)
        ## not_entailment
        df_train_step_not_entail = df_train[df_train.label_text != label_text].copy(deep=True)
        df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=random_seed)
        df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
        df_train_step_not_entail["label"] = [1] * len(df_train_step_not_entail)
        # append
        df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
    df_train = pd.concat(df_train_lst)
    
    # shuffle
    df_train = df_train.sample(frac=1, random_state=random_seed)
    df_train["label"] = df_train.label.apply(int)
    df_train["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df_train["label"]]

    print(f"After adding not_entailment training examples, the training data was augmented to {len(df_train)} texts.")
    print(f"Max augmentation could be: len(df_train) * 2 = {length_original_data_train*2}. It can also be lower, if there are more entail examples than not-entail for a majority class.")

    return df_train.copy(deep=True)

# Function to format the test set for NLI
def format_nli_testset(df_test, hypo_label_dic):
    ## explode test dataset for N hypotheses
    hypothesis_lst = [value for key, value in hypo_label_dic.items()]
    print("Number of hypotheses/classes: ", len(hypothesis_lst))

    # label lists with 0 at position of their true hypo, 1 for not-true hypos
    label_text_label_dic_explode = {}
    for key, value in hypo_label_dic.items():
        label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
        label_text_label_dic_explode[key] = label_lst

    df_test["label"] = df_test.label_text.map(label_text_label_dic_explode)
    df_test["hypothesis"] = [hypothesis_lst] * len(df_test)
    print(f"Original test set size: {len(df_test)}")
    
    # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
    df_test = df_test.explode(["hypothesis", "label"])
    print(f"Test set size for NLI classification: {len(df_test)}\n")

    df_test["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df_test["label"]]

    return df_test.copy(deep=True)

# Function to compute metrics for NLI binary classification
def compute_metrics_nli_binary(eval_pred, label_text_alphabetical):
    predictions, labels = eval_pred

    # Split in chunks with predictions for each hypothesis for one unique premise
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # For each chunk/premise, select the most likely hypothesis
    prediction_chunks_lst = list(chunks(predictions, len(label_text_alphabetical)))
    hypo_position_highest_prob = []
    for chunk in prediction_chunks_lst:
        hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))

    label_chunks_lst = list(chunks(labels, len(label_text_alphabetical)))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))

    # Calculate standard metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               }
    return metrics

# File to save results
results_file = "results_nli.txt"

# Open the results file in append mode
with open(results_file, "a") as f_results:
    # Loop over datasets
    for dataset_name in datasets_list:
        # Get hypotheses for the current dataset
        hypo_label_dic = hypotheses_per_dataset[dataset_name]
        # Determine the number of labels based on the dataset
        num_labels = len(hypo_label_dic)

        # Create alphabetically ordered list of the original dataset classes/labels
        label_text_alphabetical = np.sort(list(hypo_label_dic.keys()))

        # Load data
        train_path = dataset_paths[dataset_name]["train"]
        test_path = dataset_paths[dataset_name]["test"]
        index_col = dataset_paths[dataset_name].get("index_col", None)
        if index_col:
            df_train_full = pd.read_csv(train_path, index_col=index_col)
            df_test_full = pd.read_csv(test_path, index_col=index_col)
        else:
            df_train_full = pd.read_csv(train_path)
            df_test_full = pd.read_csv(test_path)

        total_train_size = len(df_train_full)
        print(f"Loaded data for {dataset_name}. Total train size: {total_train_size}, Test size: {len(df_test_full)}.")

        # Adjust sample sizes for the current dataset
        adjusted_sample_sizes = []
        for sample_size in sample_sizes_list:
            if sample_size == 'full' or sample_size <= total_train_size:
                adjusted_sample_sizes.append(sample_size)
            else:
                print(f"Skipping sample size {sample_size} for dataset {dataset_name} as it exceeds training data size {total_train_size}.")

        # Loop over context usage
        for use_context in use_context_options:
            context_str = "with_context" if use_context else "without_context"
            # Loop over sample sizes
            for sample_size in adjusted_sample_sizes:
                # Loop over random seeds
                for SEED_GLOBAL in seeds_list:
                    print(f"Starting iteration with dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}, {context_str}")
                    # Clean GPU memory before each iteration
                    clean_memory()

                    # Set random seed for reproducibility
                    np.random.seed(SEED_GLOBAL)
                    torch.manual_seed(SEED_GLOBAL)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED_GLOBAL)

                    # Sample training data
                    if sample_size != 'full':
                        df_train = df_train_full.sample(n=int(sample_size), random_state=SEED_GLOBAL).copy(deep=True)
                        print(f"Sampled training data to size: {len(df_train)}.")
                    else:
                        df_train = df_train_full.copy(deep=True)
                        print("Using full training data.")

                    df_test = df_test_full.copy(deep=True)

                    # Preprocess data
                    df_train, df_test = preprocess_data(df_train, df_test, dataset_name, use_context)

                    # Format data into NLI format
                    df_train_formatted = format_nli_trainset(df_train, hypo_label_dic, random_seed=SEED_GLOBAL)
                    df_test_formatted = format_nli_testset(df_test, hypo_label_dic)

                    # Convert pandas dataframes to Hugging Face dataset object
                    dataset = datasets.DatasetDict({
                        "train": datasets.Dataset.from_pandas(df_train_formatted),
                        "test": datasets.Dataset.from_pandas(df_test_formatted)
                    })

                    # Tokenize
                    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)

                    # Use GPU (cuda) if available, otherwise use CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Device: {device}")
                    model.to(device);

                    # Tokenize function for NLI format
                    def tokenize_nli_format(examples):
                        return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=512)

                    dataset["train"] = dataset["train"].map(tokenize_nli_format, batched=True)
                    dataset["test"] = dataset["test"].map(tokenize_nli_format, batched=True)

                    # Training arguments
                    training_directory = f"NLI-{dataset_name}-{context_str}-{sample_size}-{SEED_GLOBAL}"
                    train_args = TrainingArguments(
                        output_dir=f'./results/{training_directory}',
                        logging_dir=f'./logs/{training_directory}',
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=80,
                        num_train_epochs=3,
                        warmup_ratio=0.25,
                        weight_decay=0.1,
                        seed=SEED_GLOBAL,
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy_not_b",
                        fp16=False,  # Set to False as per original script
                        fp16_full_eval=False,
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
                        compute_metrics=lambda eval_pred: compute_metrics_nli_binary(eval_pred, label_text_alphabetical)
                    )

                    # Train the model
                    trainer.train()

                    # Evaluate the model
                    results = trainer.evaluate()

                    # Print and save the results
                    print(f"Results for dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}, {context_str}")
                    print(results)
                    # Save the results to the file
                    f_results.write(f"Dataset: {dataset_name}, Context: {context_str}, Sample size: {sample_size}, Seed: {SEED_GLOBAL}\n")
                    f_results.write(f"Results: {results}\n")
                    f_results.write("\n")
                    f_results.flush()  # Ensure data is written to file

                    # Clean up
                    del model, tokenizer, trainer, train_args, dataset, df_train, df_test, df_train_formatted, df_test_formatted
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
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, classification_report
import gc

# Define datasets, sample sizes, random seeds, and context usage
datasets_list = ["Hate", "Corona", "Military", "Morality"]
sample_sizes_list = [100, 500, 1000, 2500, 5000, 10000, 25000, 'full']
seeds_list = [42, 43, 44, 45, 46]
use_context_options = [True, False]  # Run with and without context

# Mapping from dataset names to file paths and index columns
dataset_paths = {
    "Morality": {
        "train": "Morality/df_manifesto_morality_train_w_context.csv",
        "test": "Morality/df_manifesto_morality_test_w_context.csv",
        "index_col": "idx"
    },
    "Corona": {
        "train": "Corona/df_coronanet_20220124_train_w_context.csv",
        "test": "Corona/df_coronanet_20220124_test_w_context.csv",
        "index_col": "idx"
    },
    "Military": {
        "train": "Military/df_manifesto_military_train_w_context.csv",
        "test": "Military/df_manifesto_military_test_w_context.csv",
        "index_col": "idx"
    },
    "Hate": {
        "train": "Hate/df_hate_train_w_context.csv",
        "test": "Hate/df_hate_test_w_context.csv",
        "index_col": None  # No 'idx' column in Hate dataset
    },
}

# Hypotheses for each dataset
hypotheses_per_dataset = {
    "Hate": {
        "Offensive Language": "The quote contains offensive language, such as curse words that are not targeted at a specific group",
        "Hate Speech": "The quote contains hate speech, language that is used to express hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group",
        "Neither": "The quote does not contain hate speech or offensive language, such as hate towards a specific group or curse words"
    },
    "Corona": {
        'Anti-Disinformation Measures': "The quote is about measures against disinformation: Efforts by the government to limit the spread of false, inaccurate or harmful information",
        'COVID-19 Vaccines': "The quote is about COVID-19 vaccines. A policy regarding the research and development, or regulation, or production, or purchase, or distribution of a vaccine.",
        'Closure and Regulation of Schools': "The quote is about regulating schools and educational establishments. For example closing an educational institution, or allowing educational institutions to open with or without certain conditions.",
        'Curfew': "The quote is about a curfew: Domestic freedom of movement is limited during certain times of the day",
        'Declaration of Emergency': "The quote is about declaration of a state of national emergency",
        'External Border Restrictions': "The quote is about external border restrictions: The ability to enter or exit country borders is reduced.",
        'Health Monitoring': "The quote is about health monitoring of individuals who are likely to be infected.",
        'Health Resources': "The quote is about health resources: For example medical equipment, number of hospitals, health infrastructure, personnel (e.g. doctors, nurses), mask purchases",
        'Health Testing': "The quote is about health testing of large populations regardless of their likelihood of being infected.",
        'Hygiene': "The quote is about hygiene: Promotion of hygiene in public spaces, for example disinfection in subways or burials.",
        'Internal Border Restrictions': "The quote is about internal border restrictions: The ability to move freely within the borders of a country is reduced.",
        'Lockdown': "The quote is about a lockdown: People are obliged to shelter in place and are only allowed to leave their shelter for specific reasons",
        'New Task Force, Bureau or Administrative Configuration': "The quote is about a new administrative body, for example a new task force, bureau or administrative configuration.",
        'Public Awareness Measures': "The quote is about public awareness measures or efforts to disseminate or gather reliable information, for example information on health prevention.",
        'Quarantine': "The quote is about quarantine. People are obliged to isolate themselves if they are infected.",
        'Restriction and Regulation of Businesses': "The quote is about restricting or regulating businesses, private commercial activities: For example closing down commercial establishments, or allowing commercial establishments to open with or without certain conditions.",
        'Restriction and Regulation of Government Services': "The quote is about restricting or regulating government services or public facilities: For example closing down government services, or allowing government services to operate with or without certain conditions.",
        'Restrictions of Mass Gatherings': "The quote is about restrictions of mass gatherings: The number of people allowed to congregate in a place is limited",
        'Social Distancing': "The quote is about social distancing, reducing contact between individuals in public spaces, mask wearing.",
        "Other Policy Not Listed Above": "The quote is about something other than regulation of businesses, government, gatherings, distancing, quarantine, lockdown, curfew, emergency, vaccines, disinformation, schools, borders or travel, testing, health resources. It is not about any of these topics."
    },
    "Morality": {
        "Traditional Morality: Positive": "The quote is positive towards traditional morality, for example in favour of traditional family values, religious institutions, or against unseemly behaviour",
        "Traditional Morality: Negative": "The quote is negative towards traditional morality, for example in favour of divorce or abortion, modern families, separation of church and state, modern values",
        "Other": "The quote is not about traditional morality, for example not about family values, abortion or religion"
    },
    "Military": {
        "Military: Positive": "The quote is positive towards the military, for example for military spending, defense, military treaty obligations.",
        "Military: Negative": "The quote is negative towards the military, for example against military spending, for disarmament, against conscription.",
        "Other": "The quote is not about military or defense"
    }
}

# Function to preprocess data based on the dataset and context usage
def preprocess_data(df_train, df_test, dataset_name, use_context):
    if dataset_name in ["Military", "Morality"]:
        if use_context:
            # With context
            df_train["text_prepared"] = (
                df_train.text_preceding.fillna("") +
                '. The quote: "' + df_train.text_original.fillna("") +
                '" - end of the quote. ' + df_train.text_following.fillna("") +
                df_train.context.fillna("")
            )
            df_test["text_prepared"] = (
                df_test.text_preceding.fillna("") +
                '. The quote: "' + df_test.text_original.fillna("") +
                '" - end of the quote. ' + df_test.text_following.fillna("") +
                df_test.context.fillna("")
            )
        else:
            # Without context
            df_train["text_prepared"] = (
                df_train.text_preceding.fillna("") +
                '. The quote: "' + df_train.text_original.fillna("") +
                '" - end of the quote. ' + df_train.text_following.fillna("")
            )
            df_test["text_prepared"] = (
                df_test.text_preceding.fillna("") +
                '. The quote: "' + df_test.text_original.fillna("") +
                '" - end of the quote. ' + df_test.text_following.fillna("")
            )
    elif dataset_name in ["Corona", "Hate"]:
        if use_context:
            # With context
            df_train["text_prepared"] = df_train.text.fillna("") + " " + df_train.context.fillna("")
            df_test["text_prepared"] = df_test.text.fillna("") + " " + df_test.context.fillna("")
        else:
            # Without context
            df_train["text_prepared"] = df_train.text.fillna("")
            df_test["text_prepared"] = df_test.text.fillna("")
    # Remove '<pad>' tokens
    df_train['text_prepared'] = df_train['text_prepared'].str.replace('<pad>', '', regex=False)
    df_test['text_prepared'] = df_test['text_prepared'].str.replace('<pad>', '', regex=False)
    return df_train, df_test

# Function to clean GPU memory
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Function to format the training set for NLI
def format_nli_trainset(df_train, hypo_label_dic, random_seed=42):
    print(f"Length of df_train before formatting step: {len(df_train)}.")
    length_original_data_train = len(df_train)

    df_train_lst = []
    for label_text, hypothesis in hypo_label_dic.items():
        ## entailment
        df_train_step = df_train[df_train.label_text == label_text].copy(deep=True)
        df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
        df_train_step["label"] = [0] * len(df_train_step)
        ## not_entailment
        df_train_step_not_entail = df_train[df_train.label_text != label_text].copy(deep=True)
        df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=random_seed)
        df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
        df_train_step_not_entail["label"] = [1] * len(df_train_step_not_entail)
        # append
        df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
    df_train = pd.concat(df_train_lst)
    
    # shuffle
    df_train = df_train.sample(frac=1, random_state=random_seed)
    df_train["label"] = df_train.label.apply(int)
    df_train["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df_train["label"]]

    print(f"After adding not_entailment training examples, the training data was augmented to {len(df_train)} texts.")
    print(f"Max augmentation could be: len(df_train) * 2 = {length_original_data_train*2}. It can also be lower, if there are more entail examples than not-entail for a majority class.")

    return df_train.copy(deep=True)

# Function to format the test set for NLI
def format_nli_testset(df_test, hypo_label_dic):
    ## explode test dataset for N hypotheses
    hypothesis_lst = [value for key, value in hypo_label_dic.items()]
    print("Number of hypotheses/classes: ", len(hypothesis_lst))

    # label lists with 0 at position of their true hypo, 1 for not-true hypos
    label_text_label_dic_explode = {}
    for key, value in hypo_label_dic.items():
        label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
        label_text_label_dic_explode[key] = label_lst

    df_test["label"] = df_test.label_text.map(label_text_label_dic_explode)
    df_test["hypothesis"] = [hypothesis_lst] * len(df_test)
    print(f"Original test set size: {len(df_test)}")
    
    # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
    df_test = df_test.explode(["hypothesis", "label"])
    print(f"Test set size for NLI classification: {len(df_test)}\n")

    df_test["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df_test["label"]]

    return df_test.copy(deep=True)

# Function to compute metrics for NLI binary classification
def compute_metrics_nli_binary(eval_pred, label_text_alphabetical):
    predictions, labels = eval_pred

    # Split in chunks with predictions for each hypothesis for one unique premise
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # For each chunk/premise, select the most likely hypothesis
    prediction_chunks_lst = list(chunks(predictions, len(label_text_alphabetical)))
    hypo_position_highest_prob = []
    for chunk in prediction_chunks_lst:
        hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))

    label_chunks_lst = list(chunks(labels, len(label_text_alphabetical)))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))

    # Calculate standard metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               }
    return metrics

# File to save results
results_file = "results_nli.txt"

# Open the results file in append mode
with open(results_file, "a") as f_results:
    # Loop over datasets
    for dataset_name in datasets_list:
        # Get hypotheses for the current dataset
        hypo_label_dic = hypotheses_per_dataset[dataset_name]
        # Determine the number of labels based on the dataset
        num_labels = len(hypo_label_dic)

        # Create alphabetically ordered list of the original dataset classes/labels
        label_text_alphabetical = np.sort(list(hypo_label_dic.keys()))

        # Load data
        train_path = dataset_paths[dataset_name]["train"]
        test_path = dataset_paths[dataset_name]["test"]
        index_col = dataset_paths[dataset_name].get("index_col", None)
        if index_col:
            df_train_full = pd.read_csv(train_path, index_col=index_col)
            df_test_full = pd.read_csv(test_path, index_col=index_col)
        else:
            df_train_full = pd.read_csv(train_path)
            df_test_full = pd.read_csv(test_path)

        total_train_size = len(df_train_full)
        print(f"Loaded data for {dataset_name}. Total train size: {total_train_size}, Test size: {len(df_test_full)}.")

        # Adjust sample sizes for the current dataset
        adjusted_sample_sizes = []
        for sample_size in sample_sizes_list:
            if sample_size == 'full' or sample_size <= total_train_size:
                adjusted_sample_sizes.append(sample_size)
            else:
                print(f"Skipping sample size {sample_size} for dataset {dataset_name} as it exceeds training data size {total_train_size}.")

        # Loop over context usage
        for use_context in use_context_options:
            context_str = "with_context" if use_context else "without_context"
            # Loop over sample sizes
            for sample_size in adjusted_sample_sizes:
                # Loop over random seeds
                for SEED_GLOBAL in seeds_list:
                    print(f"Starting iteration with dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}, {context_str}")
                    # Clean GPU memory before each iteration
                    clean_memory()

                    # Set random seed for reproducibility
                    np.random.seed(SEED_GLOBAL)
                    torch.manual_seed(SEED_GLOBAL)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED_GLOBAL)

                    # Sample training data
                    if sample_size != 'full':
                        df_train = df_train_full.sample(n=int(sample_size), random_state=SEED_GLOBAL).copy(deep=True)
                        print(f"Sampled training data to size: {len(df_train)}.")
                    else:
                        df_train = df_train_full.copy(deep=True)
                        print("Using full training data.")

                    df_test = df_test_full.copy(deep=True)

                    # Preprocess data
                    df_train, df_test = preprocess_data(df_train, df_test, dataset_name, use_context)

                    # Format data into NLI format
                    df_train_formatted = format_nli_trainset(df_train, hypo_label_dic, random_seed=SEED_GLOBAL)
                    df_test_formatted = format_nli_testset(df_test, hypo_label_dic)

                    # Convert pandas dataframes to Hugging Face dataset object
                    dataset = datasets.DatasetDict({
                        "train": datasets.Dataset.from_pandas(df_train_formatted),
                        "test": datasets.Dataset.from_pandas(df_test_formatted)
                    })

                    # Tokenize
                    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)

                    # Use GPU (cuda) if available, otherwise use CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Device: {device}")
                    model.to(device);

                    # Tokenize function for NLI format
                    def tokenize_nli_format(examples):
                        return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=512)

                    dataset["train"] = dataset["train"].map(tokenize_nli_format, batched=True)
                    dataset["test"] = dataset["test"].map(tokenize_nli_format, batched=True)

                    # Training arguments
                    training_directory = f"NLI-{dataset_name}-{context_str}-{sample_size}-{SEED_GLOBAL}"
                    train_args = TrainingArguments(
                        output_dir=f'./results/{training_directory}',
                        logging_dir=f'./logs/{training_directory}',
                        learning_rate=2e-5,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=80,
                        num_train_epochs=3,
                        warmup_ratio=0.25,
                        weight_decay=0.1,
                        seed=SEED_GLOBAL,
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy_not_b",
                        fp16=False,  # Set to False as per original script
                        fp16_full_eval=False,
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
                        compute_metrics=lambda eval_pred: compute_metrics_nli_binary(eval_pred, label_text_alphabetical)
                    )

                    # Train the model
                    trainer.train()

                    # Evaluate the model
                    results = trainer.evaluate()

                    # Print and save the results
                    print(f"Results for dataset: {dataset_name}, sample_size: {sample_size}, SEED_GLOBAL: {SEED_GLOBAL}, {context_str}")
                    print(results)
                    # Save the results to the file
                    f_results.write(f"Dataset: {dataset_name}, Context: {context_str}, Sample size: {sample_size}, Seed: {SEED_GLOBAL}\n")
                    f_results.write(f"Results: {results}\n")
                    f_results.write("\n")
                    f_results.flush()  # Ensure data is written to file

                    # Clean up
                    del model, tokenizer, trainer, train_args, dataset, df_train, df_test, df_train_formatted, df_test_formatted
                    clean_memory()

                    print("Iteration completed.\n")
>>>>>>> 53c9b4e0ed2869a8c28af646c60a06b6f826806d
