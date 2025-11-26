import json

class MiniLLM:
    def __init__(self):
        self.incidents = []

        with open("C:/mini_llm/data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.incidents.append({
                    "keywords": item["text"].lower().split(),   # naive keyword extraction
                    "summary": item["summary"],
                    "steps": item["steps"]
                })

    def get_response(self, text):
        text = text.lower()

        for incident in self.incidents:
            if any(k in text for k in incident["keywords"]):
                return {
                    "summary": incident["summary"],
                    "steps": "\n".join(incident["steps"])
                }

        return {"summary": "No AI suggestion available", "steps": ""}
