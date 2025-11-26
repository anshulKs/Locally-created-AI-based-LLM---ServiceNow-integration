# # tokenize_dataset.py
# from datasets import load_dataset
# from transformers import AutoTokenizer

# # Load structured JSON dataset
# dataset = load_dataset("json", data_files="C:/mini_llm/data.jsonl", split="train")

# # Load tokenizer from your existing mini LLM
# tokenizer = AutoTokenizer.from_pretrained("./mini_llm_model")

# # Tokenize each example: summary + steps
# # Inside the tokenize() function
# def tokenize(example):
#     text = example['summary'] + " Steps: " + example['steps']
#     tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128)
#     # Add labels for Trainer
#     tokens["labels"] = tokens["input_ids"].copy()
#     return tokens

# tokenized_dataset = dataset.map(tokenize, batched=False)

# # Save tokenized dataset
# tokenized_dataset.save_to_disk("./tokenized_dataset")

# print("Tokenization complete. Sample tokenized entry:")
# print(tokenized_dataset[0])




from datasets import load_dataset
from transformers import AutoTokenizer

# 1️⃣ Load your local text file
dataset = load_dataset("text", data_files="C:/mini_llm/data.jsonl")
print("Dataset loaded:")
print(dataset)

# 2️⃣ Load the tokenizer (small GPT model)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# 3️⃣ Tokenize each line
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized!")

# 4️⃣ Print a sample
print("Sample tokenized entry:")
print(tokenized_datasets["train"][0])
