"""Routes for Text-to-Speech functionality."""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from services.tts_service import TTSService

logger = logging.getLogger(__name__)

router = APIRouter()

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_file: str

@router.post("/tts", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    Generate speech from text using Piper TTS.
    
    Args:
        request: TTSRequest containing the text to convert
        
    Returns:
        TTSResponse with the audio filename
    """
    logger.info("=" * 80)
    logger.info("🔊 NEW TTS REQUEST")
    logger.info(f"Text length: {len(request.text)} characters")
    
    if not request.text or not request.text.strip():
        logger.error("✗ Empty text provided")
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Generate speech
        audio_filename = await TTSService.generate_speech(request.text)
        
        # Cleanup old files
        TTSService.cleanup_old_files(keep_count=10)
        
        logger.info(f"✓ TTS request completed successfully")
        logger.info("=" * 80)
        
        return TTSResponse(audio_file=audio_filename)
        
    except FileNotFoundError as e:
        logger.error(f"✗ Configuration error: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"✗ Error generating TTS: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@router.get("/tts/audio/{filename}")
async def get_audio(filename: str):
    """
    Retrieve a generated audio file.
    
    Args:
        filename: Name of the audio file to retrieve
        
    Returns:
        FileResponse with the audio file
    """
    logger.info(f"📥 Audio file requested: {filename}")
    
    # Validate filename format (security check)
    if not filename.startswith("speech_") or not filename.endswith(".wav"):
        logger.error(f"✗ Invalid filename format: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename format")
    
    try:
        audio_path = TTSService.get_audio_path(filename)
        
        if not audio_path.exists():
            logger.error(f"✗ Audio file not found: {filename}")
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        logger.info(f"✓ Serving audio file: {filename}")
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving audio file: {str(e)}")
