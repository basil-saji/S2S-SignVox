"""
tts.py - Text-to-Speech engine using Piper TTS with gTTS fallback.
"""

import io
import wave
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Wraps Piper TTS for fast, offline speech synthesis.
    Falls back to gTTS (Google Text-to-Speech) if Piper is unavailable.
    """

    def __init__(self, model_path: str = None):
        self.engine = None
        self.mode = None

        # Try loading Piper TTS first
        if model_path:
            try:
                from piper import PiperVoice
                self.voice = PiperVoice.load(model_path)
                self.mode = "piper"
                logger.info(f"Piper TTS loaded from {model_path}")
                return
            except ImportError:
                logger.warning("piper-tts not installed, trying gTTS fallback")
            except Exception as e:
                logger.warning(f"Failed to load Piper model: {e}")

        # Fallback to gTTS
        try:
            from gtts import gTTS
            self.mode = "gtts"
            logger.info("Using gTTS as TTS engine (requires internet)")
        except ImportError:
            logger.error("Neither piper-tts nor gTTS is installed!")
            self.mode = None

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to WAV audio bytes.

        Args:
            text: The sentence to convert to speech.

        Returns:
            Raw WAV file bytes.
        """
        if self.mode == "piper":
            return self._synthesize_piper(text)
        elif self.mode == "gtts":
            return self._synthesize_gtts(text)
        else:
            raise RuntimeError("No TTS engine available")

    def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using Piper TTS (offline, fast)."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize(text, wav_file)
        return buffer.getvalue()

    def _synthesize_gtts(self, text: str) -> bytes:
        """Synthesize using Google TTS (online fallback)."""
        from gtts import gTTS

        # gTTS outputs MP3, we return it as-is (Flutter can play MP3)
        buffer = io.BytesIO()
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.getvalue()
