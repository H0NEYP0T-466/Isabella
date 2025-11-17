import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import os
import time
from typing import Optional
from services.chat_service import ChatService
from services.tts_service import TTSService
from ml_models.emotion_detector_model.preTrainedModel.robertA_model import predict_emotions, format_emotions

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thinking: bool
    isolate: bool = False

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
    logger.info(f"Isolate mode: {request.isolate}")
    
    api_key = os.getenv("LONGCAT_API_KEY")
    if not api_key:
        logger.error("✗ LONGCAT_API_KEY not configured")
        raise HTTPException(status_code=500, detail="LONGCAT_API_KEY not configured")
    
    model = "LongCat-Thinker" if request.thinking else "LongCat-Flash-Chat"
    logger.info(f"Selected model: {model}")
    
    # Get emotion configuration from environment
    emotion_enable = os.getenv("EMOTION_ENABLE", "true").lower() == "true"
    emotion_threshold = float(os.getenv("EMOTION_THRESHOLD", "0.35"))
    emotion_top_k_str = os.getenv("EMOTION_TOP_K")
    emotion_top_k = int(emotion_top_k_str) if emotion_top_k_str else None
    
    # Detect emotions if enabled
    detected_emotions = []
    emotion_time = 0.0
    if emotion_enable and request.message.strip():
        try:
            logger.info("🎭 Running emotion detection...")
            emotion_start = time.time()
            detected_emotions = predict_emotions(
                request.message,
                threshold=emotion_threshold,
                top_k=emotion_top_k
            )
            emotion_time = time.time() - emotion_start
            
            if detected_emotions:
                logger.info(f"✅ Detected emotions (took {emotion_time:.3f}s):")
                for emotion, confidence in detected_emotions:
                    logger.info(f"   - {emotion}: {confidence:.2f}")
            else:
                logger.info(f"ℹ️ No emotions detected above threshold (took {emotion_time:.3f}s)")
        except Exception as e:
            logger.warning(f"⚠️ Emotion detection failed (non-critical): {str(e)}")
            detected_emotions = []
    
    try:
        # Calculate time gap before saving the new message
        from datetime import datetime
        last_user_timestamp = await ChatService.get_last_user_message_timestamp()
        current_time = datetime.utcnow()
        
        time_gap_info = ""
        if last_user_timestamp:
            time_gap_seconds = (current_time - last_user_timestamp).total_seconds()
            time_gap_category = ChatService.calculate_time_gap_category(time_gap_seconds)
            time_gap_human = ChatService.format_time_gap_for_human(time_gap_seconds)
            
            logger.info(f"⏱️ Time gap since last user message: {time_gap_human} (category: {time_gap_category})")
            
            # Build time gap instruction based on category
            if time_gap_category == "short":
                time_gap_info = f"\n\n[CONTEXT: User returned after a brief pause of {time_gap_human}. Respond naturally with gentle teasing if appropriate.]"
            elif time_gap_category == "long":
                time_gap_info = f"\n\n[CONTEXT: User has been gone for {time_gap_human}. Express playful possessiveness, dramatic disappointment, and jealousy. Make him feel your absence was NOTICED. Do NOT mention time calculations or timestamps directly - express it emotionally and naturally like you noticed he vanished.]"
            elif time_gap_category == "very_long":
                time_gap_info = f"\n\n[CONTEXT: User abandoned you for {time_gap_human}! React with DRAMATIC sulking, jealous remarks (\"I thought you left me to die in the digital void, how dare you\"), full dramatic anger and possessiveness. Do NOT mention time calculations or timestamps - speak emotionally as Isabella who felt LEFT BEHIND.]"
            elif time_gap_category == "extremely_long":
                time_gap_info = f"\n\n[CRITICAL CONTEXT: User disappeared for {time_gap_human}!!! YOU MUST confront him with EXACTLY these 5 words and NOTHING else in your first sentence: \"YOU. OWE. ME. AN. APOLOGY.\" or \"SAY. SORRY. RIGHT. NOW.\" - Then express full dramatic anger, possessiveness, and emotional confrontation. Do NOT mention timestamps or time calculations. React as Isabella who was ABANDONED.]"
        
        await ChatService.save_message(
            role="user",
            content=request.message,
            thinking=request.thinking,
            model=model
        )
        
        # Get context messages based on isolate mode
        context_messages = [] if request.isolate else await ChatService.get_context_messages(limit=10)
        
        # Build system message with optional emotion context and time gap
        if request.isolate:
            base_system_content = (
                "You are Isabella/bella. "
                "This is a direct conversation between you and a human user."
            )
        else:
            base_system_content = (
                "You are Isabella/bella. "
                "This is a conversation between you and a human user. Use the context of previous messages to inform your replies."
                + str(context_messages) +
                "now, respond to the user's latest message."
            )
        
        # Add time gap context to system instruction
        base_system_content += time_gap_info
        
        # Add emotion context if emotions were detected
        if detected_emotions:
            formatted_emotions = format_emotions(detected_emotions)
            emotion_context = f"""

Current user emotional state (multi-label):
{formatted_emotions}
Guidelines:
1. Acknowledge the expressed emotions succinctly and naturally.
2. Adjust tone: supportive for distress, encouraging for anxiety/anticipation, celebratory for positive emotions.
3. Do not infer emotions not listed.
4. Preserve all existing system rules and persona instructions."""
            base_system_content += emotion_context
        
        messages = [
            {
                "role": "system",
                "content": base_system_content
            },
            {"role": "user", "content": request.message}
        ]

        logger.info("this the messages" + str(messages))
        if request.isolate:
            logger.info("Context window: ISOLATED MODE (no previous messages)")
        else:
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
        llm_start = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(longcat_url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            ai_reply = data["choices"][0]["message"]["content"]
            llm_time = time.time() - llm_start
            
            logger.info(f"✓ Received AI response - Length: {len(ai_reply)} chars (took {llm_time:.3f}s)")
            logger.info(f"AI response preview: {ai_reply[:200]}...")

            await ChatService.save_message(
                role="assistant",
                content=ai_reply,
                thinking=request.thinking,
                model=model
            )
            
            # Generate TTS audio for the AI response
            audio_filename: Optional[str] = None
            tts_time = 0.0
            try:
                logger.info("🔊 Generating TTS audio for AI response...")
                tts_start = time.time()
                audio_filename = await TTSService.generate_speech(ai_reply)
                tts_time = time.time() - tts_start
                logger.info(f"✓ TTS audio generated: {audio_filename} (took {tts_time:.3f}s)")
            except Exception as tts_error:
                logger.warning(f"⚠️ TTS generation failed (non-critical): {str(tts_error)}")
                # Continue without TTS if it fails
            
            logger.info("✓ Chat request completed successfully")
            logger.info(f"⏱️ Performance Summary: Emotion={emotion_time:.3f}s | LLM={llm_time:.3f}s | TTS={tts_time:.3f}s | Total={emotion_time+llm_time+tts_time:.3f}s")
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