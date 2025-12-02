from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------
# Load trained model and tokenizer
# -----------------------------
model_path = r"C:\mini_llm\mini_llm_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# -----------------------------
# Prepare a test incident
# -----------------------------
incident_text = "VPN fails to connect from home network."

# -----------------------------
# Encode input
# -----------------------------
inputs = tokenizer(incident_text, return_tensors="pt")

# -----------------------------
# Generate output
# -----------------------------
output_ids = model.generate(
    **inputs,
    max_length=100,      # maximum number of tokens to generate
    num_return_sequences=1,
    do_sample=True,      # random sampling
    temperature=0.7      # creativity
)

# -----------------------------
# Decode and print
# -----------------------------
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== Generated Summary / Steps ===")
print(generated_text)

