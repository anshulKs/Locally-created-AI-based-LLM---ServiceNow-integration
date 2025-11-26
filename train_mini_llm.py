# # train_mini_llm.py
# from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
# from datasets import load_from_disk

# # Load tokenized dataset
# tokenized_dataset = load_from_disk("./tokenized_dataset")

# # Load your local mini LLM
# model = AutoModelForCausalLM.from_pretrained("./mini_llm_model")

# # Training settings
# training_args = TrainingArguments(
#     output_dir="./mini_llm_model_finetuned",
#     num_train_epochs=3,        # 1-3 is enough for small dataset
#     per_device_train_batch_size=1,
#     save_steps=100,
#     save_total_limit=2,
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )

# trainer.train()
# trainer.save_model("./mini_llm_model_finetuned")
# print("Fine-tuning complete!")






from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load tokenized dataset
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("text", data_files=r"C:\mini_llm\data.jsonl")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

# Load the small GPT model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Training arguments
training_args = TrainingArguments(
    output_dir="./mini_llm_model",
    num_train_epochs=1,          # short training for CPU
    per_device_train_batch_size=2, # small batch for low VRAM
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",            # no need for wandb or other tracking
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./mini_llm_model")
tokenizer.save_pretrained("./mini_llm_model")

print("Training complete! Model saved in ./mini_llm_model")
