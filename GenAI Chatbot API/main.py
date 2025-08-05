from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import db


app = FastAPI()
chatbot = pipeline("text-generation", model="gpt2")
db.create_table()

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question
    answer = chatbot(question, max_length=50)[0]['generated_text']
    db.add_log(question, answer)
    return {"answer": answer}

@app.get("/logs")
def get_logs():
    logs = db.get_logs()
    return {"logs": logs}
