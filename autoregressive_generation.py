<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "Nemo"

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-Nemo-Instruct-2407",
    torch_dtype=torch.bfloat16,
    cache_dir=model_path
)

model = model.to("cuda:1")
device = "cuda:1"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")

tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.padding_side='left'

import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()

df = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")

df=df_train

def generate_context_batch(texts):
    prompts = []
    bos_token = "<s>"
    eos_token = "</s>"
    system_message = """You are an explaining assistant. Can you provide some contextual information that would help classify the following text as hate speech, offensive language or none.
    Hate speech is language that is used to expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group.
    Target groups are on characteristics like race, ethnicity, gender, and sexual orientation. Do you have an indication if these groups are targeted?
    Hate speech does not include all instances of offensive language because people often use terms that are highly offensive to certain groups but in a qualitatively different manner. For example
    some African Americans often use the term n*gga in everyday language online, people use terms like h*e and b*tch when quoting rap lyrics, and teenagers use homophobic slurs as they play videogames. 
    Refrain from giving a classification yourself, just give me the context based on these considertations and aim for approximately 200 words. Here is the text:"""

    # Ensure special tokens are recognized by the tokenizer
    special_tokens = {'additional_special_tokens': ['[INST]', '[/INST]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    for text in texts:
        # Construct the prompt according to the tokenizer's expected format
        prompt = f"{bos_token}[INST]{system_message}\n\n{text}[/INST]\n"
        prompts.append(prompt)

    # Tokenize and encode the prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1512,
        add_special_tokens=True
    ).to(model.device)

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated texts without skipping special tokens
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    # Extract responses
    responses = []
    for prompt, generated_text in zip(prompts, generated_texts):
        # Use regular expression to extract the assistant's response
        match = re.search(r'\[\/INST\](.*)', generated_text, re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            # If the pattern is not found, extract text after the prompt length
            response = generated_text[len(prompt):].strip()
        responses.append(response)

    return responses

# Set batch size
batch_size = 12

# Prepare a list to collect contexts
contexts = []

# Process the data in batches
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['tweet'].iloc[i:i + batch_size].tolist()
    batch_contexts = generate_context_batch(batch_texts)
    contexts.extend(batch_contexts)

# Add the generated contexts to your DataFrame
df['context'] = contexts

# Save the updated DataFrame to a new CSV file
df.to_csv("df_hate_all_w_context.csv", index=False)

=======
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "Nemo"

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-Nemo-Instruct-2407",
    torch_dtype=torch.bfloat16,
    cache_dir=model_path
)

model = model.to("cuda:1")
device = "cuda:1"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")

tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.padding_side='left'

import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()

df = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")

df=df_train

def generate_context_batch(texts):
    prompts = []
    bos_token = "<s>"
    eos_token = "</s>"
    system_message = """You are an explaining assistant. Can you provide some contextual information that would help classify the following text as hate speech, offensive language or none.
    Hate speech is language that is used to expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group.
    Target groups are on characteristics like race, ethnicity, gender, and sexual orientation. Do you have an indication if these groups are targeted?
    Hate speech does not include all instances of offensive language because people often use terms that are highly offensive to certain groups but in a qualitatively different manner. For example
    some African Americans often use the term n*gga in everyday language online, people use terms like h*e and b*tch when quoting rap lyrics, and teenagers use homophobic slurs as they play videogames. 
    Refrain from giving a classification yourself, just give me the context based on these considertations and aim for approximately 200 words. Here is the text:"""

    # Ensure special tokens are recognized by the tokenizer
    special_tokens = {'additional_special_tokens': ['[INST]', '[/INST]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    for text in texts:
        # Construct the prompt according to the tokenizer's expected format
        prompt = f"{bos_token}[INST]{system_message}\n\n{text}[/INST]\n"
        prompts.append(prompt)

    # Tokenize and encode the prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1512,
        add_special_tokens=True
    ).to(model.device)

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated texts without skipping special tokens
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    # Extract responses
    responses = []
    for prompt, generated_text in zip(prompts, generated_texts):
        # Use regular expression to extract the assistant's response
        match = re.search(r'\[\/INST\](.*)', generated_text, re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            # If the pattern is not found, extract text after the prompt length
            response = generated_text[len(prompt):].strip()
        responses.append(response)

    return responses

# Set batch size
batch_size = 12

# Prepare a list to collect contexts
contexts = []

# Process the data in batches
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['tweet'].iloc[i:i + batch_size].tolist()
    batch_contexts = generate_context_batch(batch_texts)
    contexts.extend(batch_contexts)

# Add the generated contexts to your DataFrame
df['context'] = contexts

# Save the updated DataFrame to a new CSV file
df.to_csv("df_hate_all_w_context.csv", index=False)

>>>>>>> 53c9b4e0ed2869a8c28af646c60a06b6f826806d
print("Context added to CSV file successfully.")