from fastapi import FastAPI
from pydantic import BaseModel
from mini_llm import MiniLLM

app = FastAPI()
mini = MiniLLM()

# Define input model
class Prompt(BaseModel):
    text: str

# API endpoint
@app.post("/ai")
def ai_endpoint(prompt: Prompt):
    return mini.get_response(prompt.text)
