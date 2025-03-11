from typing import Optional
from fastapi import FastAPI
from main import app

app = FastAPI(
    title="Chatbot Backend",
    description="This is a simple chatbot backend",
)


@app.get("/")
def health_check():
    return {"Hello": "World"}
