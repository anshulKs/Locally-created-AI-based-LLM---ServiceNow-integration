from datasets import load_dataset
from transformers import AutoTokenizer

# 1️⃣ Load your dataset
dataset = load_dataset("text", data_files=r"C:\mini_llm\data.txt")
print("Dataset loaded:")
print(dataset)

# 2️⃣ Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# ✅ Set a pad token (use the EOS token if none exists)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3️⃣ Tokenize each line
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized!")

# 4️⃣ Print a sample
print("Sample tokenized entry:")
print(tokenized_datasets["train"][0])
