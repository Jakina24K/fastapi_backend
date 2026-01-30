from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import nlp

app = FastAPI(title="NLP Spam Detection API")

# CORS pour dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour dev seulement
    allow_methods=["*"],
    allow_headers=["*"],
)

# inclure le router
app.include_router(nlp.router)
