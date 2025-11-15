import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from config.database import Database
from routes.chat import router as chat_router
from routes.tts import router as tts_router
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Isabella AI Chatbot Backend")
    logger.info("=" * 80)
    await Database.connect_db()
    logger.info("=" * 80)
    yield
    logger.info("=" * 80)
    logger.info("🛑 Shutting down Isabella AI Chatbot Backend")
    await Database.close_db()
    logger.info("=" * 80)

app = FastAPI(title="Isabella AI Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(tts_router)

@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("📍 Root endpoint accessed")
    return {"message": "Isabella AI Chatbot Backend", "status": "running"}
