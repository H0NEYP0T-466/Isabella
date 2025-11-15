"""Service layer for Text-to-Speech operations using Piper TTS."""
import logging
import subprocess
from pathlib import Path
import uuid
import re

logger = logging.getLogger(__name__)

class TTSService:
    """Service for generating speech using Piper TTS."""
    
    PIPER_DIR = Path(__file__).parent.parent / "piper_tts"
    PIPER_BIN = PIPER_DIR / "piper" / "piper.exe"
    VOICE_MODEL = PIPER_DIR / "en_US-amy-medium.onnx"
    OUTPUT_DIR = PIPER_DIR / "output"
    
    SPEECH_SPEED = 0.8  # Adjust this value to control speed
    
    @classmethod
    def setup(cls):
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def clean_text_for_tts(cls, text: str) -> str:
        """
        Clean text by removing markdown formatting, emojis, and special characters.
        
        Args:
            text: Raw text that may contain markdown and emojis
            
        Returns:
            str: Cleaned text suitable for TTS
        """
        # Remove code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code (`...`)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove markdown headers (# ## ###, etc.)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold (**text** or __text__)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic (*text* or _text_)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove strikethrough (~~text~~)
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        
        # Remove links but keep the text [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove images ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove emojis (Unicode emoji ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove bullet points and list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules (---, ***, ___)
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove blockquotes (>)
        text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)
        
        # Remove remaining asterisks
        text = re.sub(r'\*', '', text)
        
        # Replace literal \n with spaces (IMPORTANT: do this before removing actual newlines)
        text = text.replace('\\n', ' ')
        
        # Replace literal \r with spaces
        text = text.replace('\\r', ' ')
        
        # Replace literal \t with spaces
        text = text.replace('\\t', ' ')
        
        # Remove actual newline characters
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)  # Remove multiple spaces
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @classmethod
    async def generate_speech(cls, text: str, speed: float = None) -> str:
        """
        Generate speech from text using Piper TTS.
        
        Args:
            text: The text to convert to speech
            speed: Optional speed multiplier (lower = faster, higher = slower)
                   If not provided, uses cls.SPEECH_SPEED
            
        Returns:
            str: Filename of the generated audio file
            
        Raises:
            Exception: If TTS generation fails
        """
        try:
            cls.setup()
            
            if not cls.PIPER_BIN.exists():
                raise FileNotFoundError(
                    f"Piper binary not found at {cls.PIPER_BIN}. "
                    "Please install Piper TTS as described in piper_tts/README.md"
                )
            
            if not cls.VOICE_MODEL.exists():
                raise FileNotFoundError(
                    f"Voice model not found at {cls.VOICE_MODEL}. "
                    "Please download en_US-amy-medium voice model as described in piper_tts/README.md"
                )
            
            # Clean the text before TTS processing
            cleaned_text = cls.clean_text_for_tts(text)
            
            if not cleaned_text:
                raise ValueError("Text is empty after cleaning")
            
            audio_filename = f"speech_{uuid.uuid4()}.wav"
            audio_path = cls.OUTPUT_DIR / audio_filename
            
            # Use provided speed or default
            speech_speed = speed if speed is not None else cls.SPEECH_SPEED
            
            logger.info(f"🔊 Generating TTS for text (original length: {len(text)}, cleaned length: {len(cleaned_text)} chars)")
            logger.info(f"Speech speed: {speech_speed} (lower = faster)")
            logger.info(f"Using voice model: {cls.VOICE_MODEL}")
            logger.info(f"Output file: {audio_path}")

            process = subprocess.Popen(
                [
                    str(cls.PIPER_BIN),
                    "--model", str(cls.VOICE_MODEL),
                    "--output_file", str(audio_path),
                    "--length_scale", str(speech_speed)  # Control speech speed
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"  # Replace unencodable characters rather than crashing
            )
            
            stdout, stderr = process.communicate(input=cleaned_text, timeout=60)
            
            if process.returncode != 0:
                logger.error(f"✗ Piper TTS failed with return code {process.returncode}")
                logger.error(f"stderr: {stderr}")
                raise Exception(f"Piper TTS failed: {stderr.strip()}")
            
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
            
            for audio_file in audio_files[keep_count:]:
                audio_file.unlink()
                logger.info(f"🗑️ Cleaned up old TTS file: {audio_file.name}")
                
        except Exception as e:
            logger.error(f"✗ Error during cleanup: {str(e)}")