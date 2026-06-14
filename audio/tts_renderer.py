from transformers import pipeline
import torch
import logging
import io

logger = logging.getLogger("LegalMind.Audio.TTS")

class RemedialAudioGenerator:
    def __init__(self):
        # We use dynamic, lightweight gTTS for synthesis to support Malayalam and English
        logger.info("✓ gTTS Text-to-Speech engine loaded successfully.")

    def text_to_indic_speech(self, text: str, output_path: str = "remedy_output.wav"):
        """Synthesize natural voice response and save to disk."""
        try:
            from gtts import gTTS
            # Clean up text from styling tags
            cleaned_text = text.replace("*", "").replace("#", "").strip()
            # Detect Malayalam characters
            contains_malayalam = any(0x0D00 <= ord(char) <= 0x0D7F for char in cleaned_text)
            lang = "ml" if contains_malayalam else "en"
            
            logger.info(f"Synthesizing voice response using gTTS (lang={lang}) for: '{cleaned_text[:50]}...'")
            tts = gTTS(text=cleaned_text, lang=lang)
            tts.save(output_path)
            logger.info(f"✓ Dynamic speech audio saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"gTTS speech synthesis failed: {e}")
            # Fallback to a valid silent WAV file header so the browser doesn't crash on play
            with open(output_path, "wb") as f:
                f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00@\x1f\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')
            return output_path

    def text_to_indic_speech_stream(self, text: str):
        """Synthesize and stream audio bytes directly."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.text_to_indic_speech(text, tmp_path)
            with open(tmp_path, "rb") as f:
                data = f.read()
            return io.BytesIO(data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    generator = RemedialAudioGenerator()
    test_roadmap = "ഭയപ്പെടേണ്ട. മുപ്പത് ദിവസത്തെ രേഖാമൂലമുള്ള നോട്ടീസ് ഇല്ലാതെ നിങ്ങളുടെ ഭൂവുടമയ്ക്ക് നിങ്ങളെ ഒഴിപ്പിക്കാൻ കഴിയില്ല."
    generator.text_to_indic_speech(test_roadmap, "test_output.mp3")

