"""Service layer for Text-to-Speech operations using Piper TTS."""
import logging
import subprocess
import os
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class TTSService:
    """Service for generating speech using Piper TTS."""
    
    PIPER_DIR = Path(__file__).parent.parent / "piper_tts"
    PIPER_BIN = PIPER_DIR / "piper" / "piper"
    VOICE_MODEL = PIPER_DIR / "en_US-amy-medium.onnx"
    OUTPUT_DIR = PIPER_DIR / "output"
    
    @classmethod
    def setup(cls):
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    async def generate_speech(cls, text: str) -> str:
        """
        Generate speech from text using Piper TTS.
        
        Args:
            text: The text to convert to speech
            
        Returns:
            str: Path to the generated audio file
            
        Raises:
            Exception: If TTS generation fails
        """
        try:
            # Setup output directory
            cls.setup()
            
            # Check if Piper binary exists
            if not cls.PIPER_BIN.exists():
                raise FileNotFoundError(
                    f"Piper binary not found at {cls.PIPER_BIN}. "
                    "Please install Piper TTS as described in piper_tts/README.md"
                )
            
            # Check if voice model exists
            if not cls.VOICE_MODEL.exists():
                raise FileNotFoundError(
                    f"Voice model not found at {cls.VOICE_MODEL}. "
                    "Please download en_US-amy-medium voice model as described in piper_tts/README.md"
                )
            
            # Generate unique filename
            audio_filename = f"speech_{uuid.uuid4()}.wav"
            audio_path = cls.OUTPUT_DIR / audio_filename
            
            logger.info(f"🔊 Generating TTS for text (length: {len(text)} chars)")
            logger.info(f"Using voice model: {cls.VOICE_MODEL}")
            logger.info(f"Output file: {audio_path}")
            
            # Run Piper TTS
            # Command: echo "text" | piper --model model.onnx --output_file output.wav
            process = subprocess.Popen(
                [str(cls.PIPER_BIN), "--model", str(cls.VOICE_MODEL), "--output_file", str(audio_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text, timeout=30)
            
            if process.returncode != 0:
                logger.error(f"✗ Piper TTS failed with return code {process.returncode}")
                logger.error(f"stderr: {stderr}")
                raise Exception(f"Piper TTS failed: {stderr}")
            
            if not audio_path.exists():
                raise Exception("Audio file was not generated")
            
            logger.info(f"✓ TTS audio generated successfully: {audio_filename}")
            return audio_filename
            
        except subprocess.TimeoutExpired:
            logger.error("✗ Piper TTS process timed out")
            raise Exception("TTS generation timed out")
        except Exception as e:
            logger.error(f"✗ Error generating TTS: {str(e)}")
            raise
    
    @classmethod
    def get_audio_path(cls, filename: str) -> Path:
        """Get the full path to an audio file."""
        return cls.OUTPUT_DIR / filename
    
    @classmethod
    def cleanup_old_files(cls, keep_count: int = 10):
        """Remove old audio files, keeping only the most recent ones."""
        try:
            if not cls.OUTPUT_DIR.exists():
                return
            
            audio_files = sorted(
                cls.OUTPUT_DIR.glob("speech_*.wav"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove old files
            for audio_file in audio_files[keep_count:]:
                audio_file.unlink()
                logger.info(f"🗑️ Cleaned up old TTS file: {audio_file.name}")
                
        except Exception as e:
            logger.error(f"✗ Error during cleanup: {str(e)}")
