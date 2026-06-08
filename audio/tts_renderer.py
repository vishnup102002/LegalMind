from transformers import pipeline
import torch
import logging
import io

logger = logging.getLogger("LegalMind.Audio.TTS")

class RemedialAudioGenerator:
    def __init__(self):
        # Initialize sovereign Sooktam-2 model for high-fidelity regional delivery
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(
                "text-to-speech", 
                model="bharatgenai/sooktam2", 
                device=device,
                trust_remote_code=True
            )
            logger.info("✓ Sooktam-2 text-to-speech pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Sooktam-2 pipeline locally: {e}. Falling back to stub mode.")
            self.pipe = None

    def text_to_indic_speech(self, text: str, output_path: str = "remedy_output.wav"):
        """Synthesize natural voice response with accurate regional cadences and save to disk."""
        if self.pipe:
            audio_output = self.pipe(text, forward_params={"cls_language": "malayalam"})
            with open(output_path, "wb") as f:
                f.write(audio_output["audio"])
            print(f"✓ Remedial audio path rendered: {output_path}")
            return output_path
        else:
            # Stub response generation
            print(f"[STUB TTS] Synthesizing audio for text: '{text}' -> {output_path}")
            with open(output_path, "wb") as f:
                f.write(b"RIFFStubWavDataPlaceholder")
            return output_path

    def text_to_indic_speech_stream(self, text: str):
        """Synthesize and stream audio bytes directly for WebRTC connection."""
        if self.pipe:
            audio_output = self.pipe(text, forward_params={"cls_language": "malayalam"})
            return io.BytesIO(audio_output["audio"])
        else:
            # Stub stream
            return io.BytesIO(b"RIFFStubWavStreamPlaceholder")

if __name__ == "__main__":
    generator = RemedialAudioGenerator()
    test_roadmap = "ഭയപ്പെടേണ്ട. മുപ്പത് ദിവസത്തെ രേഖാമൂലമുള്ള നോട്ടീസ് ഇല്ലാതെ നിങ്ങളുടെ ഭൂവുടമയ്ക്ക് നിങ്ങളെ ഒഴിപ്പിക്കാൻ കഴിയില്ല."
    generator.text_to_indic_speech(test_roadmap)
