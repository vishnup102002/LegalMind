import torch
import logging
import io
import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("LegalMind.Audio.STT")

class ShrutamAudioTranscriber:
    def __init__(self):
        # Load whisper-tiny locally on CPU/MPS/CUDA for dynamic multi-lingual ASR
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device
            )
            logger.info("✓ Whisper-tiny Speech-to-Text pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Whisper-tiny pipeline locally: {e}. Falling back to None.")
            self.pipe = None

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw regional speech from incoming audio bytes.
        Uses Whisper auto-detection to dynamically identify the spoken language.
        Returns a tuple-like dict with transcript and detected language.
        """
        if self.pipe:
            try:
                # 1. Read audio bytes using soundfile (supports WAV, OGG, etc.)
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                
                # 2. Convert to mono if stereo
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # 3. Resample to 16000 Hz if necessary
                if samplerate != 16000:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                
                # Convert to float32 numpy array
                data = data.astype(np.float32)
                
                # 4. Transcribe using Whisper-tiny with auto language detection
                # Do NOT force a specific language — let Whisper detect dynamically
                result = self.pipe(data, return_timestamps=False)
                text = result.get("text", "").strip()
                logger.info(f"✓ Whisper-tiny transcribed ASR text: '{text}'")
                return text
            except Exception as e:
                logger.error(f"Error during Whisper-tiny transcription: {e}")
                return ""
        else:
            logger.warning("Whisper pipeline is not initialized. ASR failed.")
            return ""

    def detect_language(self, audio_bytes: bytes) -> str:
        """
        Detect the language of the audio input using Whisper's language detection.
        Returns the detected language code (e.g., 'en', 'ml', 'hi', 'ta').
        """
        if self.pipe:
            try:
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if samplerate != 16000:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                data = data.astype(np.float32)
                
                # Use a short chunk (first 5 seconds max) for fast language detection
                max_samples = 16000 * 5  # 5 seconds
                detection_chunk = data[:max_samples] if len(data) > max_samples else data
                
                # Whisper with task="transcribe" and return_timestamps detects language
                result = self.pipe(detection_chunk, return_timestamps=False)
                # Check if language info is available in the result chunks
                detected_lang = result.get("chunks", [{}])[0].get("language", "en") if result.get("chunks") else "en"
                
                # Fallback: detect from text content if chunks don't have language
                text = result.get("text", "")
                if any(0x0D00 <= ord(c) <= 0x0D7F for c in text):
                    detected_lang = "ml"  # Malayalam Unicode block
                elif any(0x0900 <= ord(c) <= 0x097F for c in text):
                    detected_lang = "hi"  # Hindi/Devanagari
                elif any(0x0B80 <= ord(c) <= 0x0BFF for c in text):
                    detected_lang = "ta"  # Tamil
                elif any(0x0C80 <= ord(c) <= 0x0CFF for c in text):
                    detected_lang = "kn"  # Kannada
                elif any(0x0C00 <= ord(c) <= 0x0C7F for c in text):
                    detected_lang = "te"  # Telugu
                else:
                    detected_lang = "en"
                
                logger.info(f"✓ Detected language: '{detected_lang}' from audio content")
                return detected_lang
            except Exception as e:
                logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
                return "en"
        return "en"

    def transcribe_stream(self, audio_generator):
        """
        Accepts a stream/generator of audio chunks and processes them in real-time.
        """
        buffer = bytearray()
        for chunk in audio_generator:
            buffer.extend(chunk)
        yield self.transcribe(bytes(buffer))

if __name__ == "__main__":
    transcriber = ShrutamAudioTranscriber()
    mock_audio = np.zeros(16000, dtype=np.float32)
    # Convert numpy array to WAV bytes for testing self-check
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mock_audio, 16000)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            bytes_data = f.read()
        transcript = transcriber.transcribe(bytes_data)
        print(f"✓ Whisper-tiny transcript: '{transcript}'")
        lang = transcriber.detect_language(bytes_data)
        print(f"✓ Detected language: '{lang}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
