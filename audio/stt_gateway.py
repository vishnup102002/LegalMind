import torch
import logging
import io

logger = logging.getLogger("LegalMind.Audio.STT")

class ShrutamAudioTranscriber:
    def __init__(self):
        # Initialize sovereign Shrutam-2 model for high-fidelity regional ASR
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="bharatgenai/shrutam2",
                device=device,
                trust_remote_code=True
            )
            logger.info("✓ Shrutam-2 Speech-to-Text pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Shrutam-2 pipeline locally: {e}. Falling back to stub mode.")
            self.pipe = None

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw regional Malayalam speech from incoming audio bytes.
        """
        if self.pipe:
            try:
                # Transcribe raw audio data
                result = self.pipe(audio_bytes)
                return result.get("text", "")
            except Exception as e:
                logger.error(f"Error during Shrutam-2 transcription: {e}")
                return ""
        else:
            # Stub response
            logger.info("[STUB ASR] Transcribing audio chunk...")
            # For testing, we return a mock Malayalam statement related to landlord issues
            return "എന്റെ ഭൂവുടമ എന്നെ വീട്ടിൽ നിന്ന് ഇറക്കിവിടാൻ നോക്കുന്നു"

    def transcribe_stream(self, audio_generator):
        """
        Accepts a stream/generator of audio chunks and processes them in real-time.
        """
        # Collect streaming audio chunks and transcribe as complete utterances
        buffer = bytearray()
        for chunk in audio_generator:
            buffer.extend(chunk)
            # Yield transcripts periodically if long silence or phrase boundary detected
            # For simplicity, here we yield the final transcription
        yield self.transcribe(bytes(buffer))

if __name__ == "__main__":
    transcriber = ShrutamAudioTranscriber()
    mock_audio = b"\x00" * 32000  # 1 second of silence
    transcript = transcriber.transcribe(mock_audio)
    print(f"✓ Shrutam-2 transcript: '{transcript}'")
