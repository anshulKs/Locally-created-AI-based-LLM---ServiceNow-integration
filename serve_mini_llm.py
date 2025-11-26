# from fastapi import FastAPI, Request
# from mini_llm import MiniLLM

# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# class GenerateRequest(BaseModel):
#     text: str

# # Load model and tokenizer
# model_path = r"C:\mini_llm\mini_llm_model"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)


# app = FastAPI()
# model = MiniLLM()

# @app.post("/generate")
# async def generate(request: GenerateRequest):
#     text = request.text  # FastAPI automatically parses JSON into this object

#     if not text:
#         return {"summary": "", "steps": ""}

#     ai_response = model.get_response(text)
#     summary = ai_response.get("summary", "")
#     steps = ai_response.get("steps", "")

#     return {"summary": summary, "steps": steps}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("serve_mini_llm:app", host="127.0.0.1", port=8000, reload=True)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = r"C:\mini_llm\mini_llm_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

app = FastAPI()

class Incident(BaseModel):
    text: str

@app.post("/generate")
def generate(incident: Incident):
    inputs = tokenizer(incident.text, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"summary": generated_text}
