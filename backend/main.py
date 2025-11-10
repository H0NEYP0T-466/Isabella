from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thinking: bool

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    api_key = os.getenv("LONGCAT_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="LONGCAT_API_KEY not configured")
    

    model = "LongCat-Thinker" if request.thinking else "LongCat-Flash-Chat"
    longcat_url = "https://api.longcat.chat/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }   
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.message}],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(longcat_url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
  
            ai_reply = data["choices"][0]["message"]["content"]
            return ChatResponse(reply=ai_reply)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Error calling LongCat API: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Unexpected response format from LongCat API: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Chatbot Backend"}
