import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import os
from typing import Optional
from services.chat_service import ChatService
from services.tts_service import TTSService

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thinking: bool

class ChatResponse(BaseModel):
    reply: str
    audio_file: Optional[str] = None  # Made optional to avoid validation error when TTS fails

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with MongoDB integration."""
    logger.info("=" * 80)
    logger.info("📨 NEW CHAT REQUEST")
    logger.info(f"User message: {request.message}")
    logger.info(f"Thinking mode: {request.thinking}")
    
    api_key = os.getenv("LONGCAT_API_KEY")
    if not api_key:
        logger.error("✗ LONGCAT_API_KEY not configured")
        raise HTTPException(status_code=500, detail="LONGCAT_API_KEY not configured")
    
    model = "LongCat-Thinker" if request.thinking else "LongCat-Flash-Chat"
    logger.info(f"Selected model: {model}")
    
    try:
        await ChatService.save_message(
            role="user",
            content=request.message,
            thinking=request.thinking,
            model=model
        )
        
        context_messages = await ChatService.get_context_messages(limit=10)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Isabella/bella."
                    "This is a conversation between you and a human user. Use the context of previous messages to inform your replies."
                    + str(context_messages) +
                    "now, respond to the user's latest message."
                )
            },
            {"role": "user", "content": request.message}
        ]

        logger.info("this the messages" + str(messages))
        logger.info(f"Context window: {len(context_messages)} previous messages")
        logger.info("Context messages:")
        for i, msg in enumerate(context_messages, 1):
            role = msg["role"]
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            logger.info(f"  [{i}] {role}: {content_preview}")

        longcat_url = "https://api.longcat.chat/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 1.0,
        }
        
        logger.info("🔄 Calling LongCat API...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(longcat_url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            ai_reply = data["choices"][0]["message"]["content"]
            
            logger.info(f"✓ Received AI response - Length: {len(ai_reply)} chars")
            logger.info(f"AI response preview: {ai_reply[:200]}...")

            await ChatService.save_message(
                role="assistant",
                content=ai_reply,
                thinking=request.thinking,
                model=model
            )
            
            # Generate TTS audio for the AI response
            audio_filename: Optional[str] = None
            try:
                logger.info("🔊 Generating TTS audio for AI response...")
                audio_filename = await TTSService.generate_speech(ai_reply)
                logger.info(f"✓ TTS audio generated: {audio_filename}")
            except Exception as tts_error:
                logger.warning(f"⚠️ TTS generation failed (non-critical): {str(tts_error)}")
                # Continue without TTS if it fails
            
            logger.info("✓ Chat request completed successfully")
            logger.info("=" * 80)
            
            return ChatResponse(reply=ai_reply, audio_file=audio_filename)
            
    except httpx.HTTPError as e:
        logger.error(f"✗ Error calling LongCat API: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Error calling LongCat API: {str(e)}")
    except (KeyError, IndexError) as e:
        logger.error(f"✗ Unexpected response format from LongCat API: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Unexpected response format from LongCat API: {str(e)}")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/messages")
async def get_messages():
    """Fetch the last 50 messages from the database."""
    try:
        logger.info("📥 Fetching recent messages from database")
        messages = await ChatService.get_recent_messages(limit=50)
        logger.info(f"✓ Returning {len(messages)} messages to client")
        return {"messages": messages}
    except Exception as e:
        logger.error(f"✗ Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")